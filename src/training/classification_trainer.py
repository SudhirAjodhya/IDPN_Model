"""
Classification trainer for Talebi & Milanfar (ICCV 2021) Table 4 reproduction.

One function, `train_one(backbone_name, config, seed, epochs, ...)`, trains a
single Table-4 row (see src/experiments/talebi_table4.py for the row list).

Design choices:
  - Config-driven: no hardcoded phase A/B. `config.init_from` names the prior
    row whose backbone checkpoint we load (Table 3 protocol: bilinear-224 acts
    as init for every proposed_* row).
  - Paper-faithful recipe: SGD momentum 0.9, weight decay 1e-4, LR 0.05 with
    exponential decay 0.94 every 2 epochs, cross-entropy with label smoothing
    0.1, HFlip-only augmentation. No early stopping (paper trains fixed epochs).
  - We save BOTH final and best-val checkpoints per config so downstream
    reporting can quote either (paper reports final; we also want best-val
    for overfit-resistant comparisons).
  - Full metric bundle per run: per-epoch train/val loss/top-1/top-5, LR,
    epoch time; confusion matrix; MACs/FLOPs/param count; config + env.
"""
from __future__ import annotations
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import torch
import torch.nn as nn

from src.data.imagenette2 import get_loaders
from src.evaluation.classification_metrics import evaluate
from src.experiments.talebi_table4 import TableRow, get_config
from src.models.classifiers.resizer_backbone import ResizerBackbone
from src.models.downsamplers.talebi import TalebiPaperResizer
from src.utils.flops import count_flops_and_params
from src.utils.misc import set_seed
from src.visualization.classification_curves import (
    save_classification_curves, save_confusion_matrix,
)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT_DIR / "data_sets" / "imagenette2" / "imagenette2"
OUTPUT_ROOT = ROOT_DIR / "outputs" / "talebi_table4"


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------

def backbone_dir(backbone: str, smoke: bool = False) -> Path:
    """Root output dir for a given backbone; suffix _smoke when smoke=True."""
    suffix = "_smoke" if smoke else ""
    return OUTPUT_ROOT / f"{backbone}{suffix}"


def ckpt_path(backbone: str, config_label: str, seed: int,
              kind: str = "final", smoke: bool = False) -> Path:
    """kind ∈ {'final', 'best'}."""
    d = backbone_dir(backbone, smoke) / "checkpoints"
    return d / f"{config_label}_seed{seed}_{kind}.pt"


def result_path(backbone: str, config_label: str, seed: int,
                smoke: bool = False) -> Path:
    d = backbone_dir(backbone, smoke) / "results"
    return d / f"{config_label}_seed{seed}.json"


def plots_dir(backbone: str, smoke: bool = False) -> Path:
    return backbone_dir(backbone, smoke) / "plots"


# ---------------------------------------------------------------------------
# resizer factory
# ---------------------------------------------------------------------------

def build_resizer(resizer_name: str | None, target_res: int) -> nn.Module | None:
    if resizer_name is None:
        return None
    if resizer_name == "talebi_paper":
        return TalebiPaperResizer(in_channels=3, n_filters=16, n_blocks=1,
                                   target_size=target_res)
    raise ValueError(f"unknown resizer {resizer_name!r}")


# ---------------------------------------------------------------------------
# environment metadata (saved once per JSON so results are self-describing)
# ---------------------------------------------------------------------------

def _env_metadata(device: torch.device) -> dict:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if device.type == "cuda":
        meta["cuda_device"] = torch.cuda.get_device_name(device)
        meta["cuda_version"] = torch.version.cuda
        meta["vram_gb"] = round(
            torch.cuda.get_device_properties(device).total_memory / 1e9, 2
        )
    return meta


# ---------------------------------------------------------------------------
# main entrypoint
# ---------------------------------------------------------------------------

def train_one(*, backbone_name: str, config: TableRow | str, seed: int,
              epochs: int, data_root: Path, device: torch.device,
              smoke: bool = False):
    """
    Train a single Table-4 row. Returns the results dict and writes it to
    outputs/talebi_table4/{backbone}/results/{config}_seed{seed}.json.

    If config.init_from is set, we load the *final* checkpoint of that prior
    config for the same (backbone, seed) and use it as the backbone init.
    """
    if isinstance(config, str):
        config = get_config(config)

    set_seed(seed)

    # smoke overrides: tiny batch, 2 epochs, minimal I/O
    if smoke:
        batch = min(config.batch_size, 8)
        epochs = 2
        num_workers = 2
    else:
        batch = config.batch_size
        num_workers = 4

    # data
    train_ld, val_ld, n_classes = get_loaders(config.input_res, batch,
                                              data_root, num_workers)
    assert n_classes == 10, \
        f"expected 10 imagenette classes at {data_root}, got {n_classes}"

    # model: [optional resizer] -> backbone
    resizer = build_resizer(config.resizer, config.target_res)
    model = ResizerBackbone(resizer, num_classes=n_classes,
                            backbone=backbone_name).to(device)

    # optional backbone init from a prior run's checkpoint
    if config.init_from is not None:
        prev = ckpt_path(backbone_name, config.init_from, seed,
                         kind="final", smoke=smoke)
        if not prev.exists():
            raise FileNotFoundError(
                f"init checkpoint missing: {prev}. "
                f"Run config={config.init_from!r} first."
            )
        state = torch.load(prev, map_location=device, weights_only=True)
        model.backbone.load_state_dict(state["backbone"])
        print(f"[{backbone_name}/{config.label}/seed{seed}] "
              f"loaded backbone from {prev.name}")

    # compute FLOPS + params BEFORE optimizer setup so no state changes leak
    perf = count_flops_and_params(
        model, input_shape=(3, config.input_res, config.input_res),
        device=device,
    )
    print(f"[{backbone_name}/{config.label}/seed{seed}] "
          f"macs={perf['macs']/1e9:.2f}G  params={perf['params']:,}")

    # paper §4 recipe
    optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda e: 0.94 ** (e // 2)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")

    epoch_logs = []
    best_val_loss = float("inf")
    best_val_top1 = 0.0
    best_epoch = -1
    tag = f"[{backbone_name}/{config.label}/seed{seed}{' SMOKE' if smoke else ''}]"

    # MLflow
    mlflow_db_path = Path("outputs/mlflow/mlruns.db").resolve().as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment("talebi_table4")
    run_name = f"{backbone_name}_{config.label}_seed{seed}{'_smoke' if smoke else ''}"

    mlflow.start_run(run_name=run_name)
    try:
        mlflow.log_params({
            "backbone": backbone_name,
            "config": config.label,
            "resizer": config.resizer or "none",
            "seed": seed,
            "epochs": epochs,
            "smoke": smoke,
            "input_res": config.input_res,
            "target_res": config.target_res,
            "batch_size": batch,
            "init_from": config.init_from or "random",
            "macs": perf["macs"],
            "params": perf["params"],
        })

        for epoch in range(epochs):
            model.train()
            t0 = time.time()
            loss_sum = c1 = total = 0

            for images, labels in train_ld:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optim.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                # gradient clipping to prevent AMP overflow / random-init explosions
                # (defensive; paper doesn't mention it but it's standard hygiene)
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optim)
                scaler.update()

                loss_sum += loss.item() * labels.size(0)
                c1 += logits.argmax(1).eq(labels).sum().item()
                total += labels.size(0)

            sched.step()
            train_top1 = 100.0 * c1 / total
            train_loss = loss_sum / total

            val = evaluate(model, val_ld, device, criterion=criterion,
                           num_classes=n_classes)
            dt = time.time() - t0

            log = {
                "epoch": epoch,
                "lr": optim.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_top1": train_top1,
                "val_loss": val["val_loss"],
                "val_top1": val["top1"],
                "val_top5": val["top5"],
                "epoch_time_s": dt,
            }
            epoch_logs.append(log)

            print(f"{tag} ep {epoch:03d}  lr={log['lr']:.4f}  "
                  f"tr_loss={train_loss:.4f} tr_top1={train_top1:.2f}  "
                  f"val_loss={val['val_loss']:.4f} val_top1={val['top1']:.2f} "
                  f"val_top5={val['top5']:.2f}  {dt:.1f}s", flush=True)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_top1": train_top1,
                "val_loss": val["val_loss"],
                "val_top1": val["top1"],
                "val_top5": val["top5"],
                "lr": optim.param_groups[0]["lr"],
            }, step=epoch)

            # track best-val (in-memory) regardless of smoke; guard only the disk save
            if val["val_loss"] is not None and val["val_loss"] < best_val_loss:
                best_val_loss = val["val_loss"]
                best_val_top1 = val["top1"]
                best_epoch = epoch
                if not smoke:
                    best_p = ckpt_path(backbone_name, config.label, seed,
                                       kind="best", smoke=smoke)
                    best_p.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "backbone": model.backbone.state_dict(),
                        "resizer": (model.resizer.state_dict()
                                    if model.resizer is not None else None),
                        "epoch": epoch,
                        "val_loss": val["val_loss"],
                        "val_top1": val["top1"],
                    }, best_p)

        # end-of-training: save FINAL checkpoint + full JSON
        final_p = ckpt_path(backbone_name, config.label, seed,
                            kind="final", smoke=smoke)
        final_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "backbone": model.backbone.state_dict(),
            "resizer": (model.resizer.state_dict()
                        if model.resizer is not None else None),
            "epoch": epochs - 1,
            "val_loss": val["val_loss"],
            "val_top1": val["top1"],
        }, final_p)

        result = {
            "backbone": backbone_name,
            "config": {
                "label": config.label,
                "resizer": config.resizer,
                "input_res": config.input_res,
                "target_res": config.target_res,
                "batch_size": batch,
                "init_from": config.init_from,
            },
            "seed": seed,
            "epochs": epochs,
            "smoke": smoke,
            # final-epoch metrics (paper-comparable)
            "final_val_top1": val["top1"],
            "final_val_top5": val["top5"],
            "final_val_loss": val["val_loss"],
            # best-val metrics (overfit-resistant)
            "best_val_top1": best_val_top1 if best_epoch >= 0 else val["top1"],
            "best_val_loss": best_val_loss if best_epoch >= 0 else val["val_loss"],
            "best_epoch": best_epoch,
            # compute characteristics
            "macs": perf["macs"],
            "flops": perf["flops"],
            "params": perf["params"],
            # full curves + confusion matrix for reconstruction
            "cm": val["cm"],
            "epoch_logs": epoch_logs,
            # environment
            "env": _env_metadata(device),
            "recipe": {
                "optimizer": "SGD momentum 0.9 wd 1e-4",
                "lr_init": 0.05,
                "lr_schedule": "exp decay 0.94 every 2 epochs (paper §4)",
                "loss": "cross-entropy with label smoothing 0.1",
                "augmentation": "HFlip only (paper §4)",
                "mixed_precision": True,
            },
        }

        # write JSON + plots + checkpoints (skip heavy I/O on smoke)
        if not smoke:
            out = result_path(backbone_name, config.label, seed, smoke=smoke)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2))
            plot_dir = plots_dir(backbone_name, smoke=smoke)
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_tag = f"{config.label}_seed{seed}"
            save_classification_curves(epoch_logs, plot_dir, phase_tag=plot_tag)
            save_confusion_matrix(val["cm"], plot_dir, phase_tag=plot_tag)
            mlflow.log_artifact(str(out), artifact_path="results")
            for name in (f"training_curves_{plot_tag}.png",
                         f"confusion_matrix_{plot_tag}.png"):
                p = plot_dir / name
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="plots")

        print(f"{tag} final: val_top1={val['top1']:.2f}  "
              f"best_val_top1={best_val_top1:.2f} @ep{best_epoch}")

    finally:
        mlflow.end_run()

    return result
