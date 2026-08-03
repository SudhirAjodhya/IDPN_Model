"""
Classification trainer for Talebi paper reproduction

Two-phase training protocol (Talebi & Milanfar, ICCV 2021):
  Phase A: Train ResNet-50 classifier with bilinear-224 input (baseline).
  Phase B: Attach TalebiPaperResizer, load Phase A backbone, train 368->224.
"""

import json
import time
import mlflow
from pathlib import Path

import torch
import torch.nn as nn

from src.data.imagenette2 import get_loaders
from src.evaluation.classification_metrics import evaluate
from src.models.classifiers.resizer_backbone import ResizerBackbone
from src.models.downsamplers.talebi import TalebiPaperResizer
from src.utils.misc import set_seed
from src.visualization.classification_curves import save_classification_curves, save_confusion_matrix


# Directory configurations
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT_DIR / "data_sets" / "imagenette2" / "imagenette2"
CKPT_DIR = ROOT_DIR / "outputs" / "talebi_reproduce" / "checkpoints"
RESULT_DIR = ROOT_DIR / "outputs" / "talebi_reproduce" / "results"


def _ckpt_path(seed: int, smoke: bool = False) -> Path:
    return CKPT_DIR / (f"phaseA_seed{seed}_smoke.pt" if smoke else f"phaseA_seed{seed}.pt")


def train_one(*, phase: str, seed: int, epochs: int, data_root: Path, device: torch.device, smoke: bool = False):
    """
    Train one phase of the Talebi reproduction experiment

    Args:
        phase: "A" (bilinear baseline) or "B" (Talebi resizer)
        seed: Random seed for reproducibility
        epochs: Number of training epochs
        data_root: Path to ImageNette dataset root
        device: Torch device
        smoke: If True, run a quick 2-epoch sanity check

    Returns:
        dict with phase results including final val top-1/top-5 and epoch logs
    """

    set_seed(seed)

    # Per-phase config (paper Table 4, ResNet-50 headline row)
    if phase == "A":
        input_res, target_res, batch = 224, 224, 128
        resizer = None
        prev_ckpt = None
    elif phase == "B":
        input_res, target_res, batch = 368, 224, 48
        resizer = TalebiPaperResizer(in_channels=3, n_filters=16, n_blocks=1, target_size=target_res)
        prev_ckpt = _ckpt_path(seed, smoke=smoke)

        if not prev_ckpt.exists():
            raise FileNotFoundError(f"Phase A checkpoint missing: {prev_ckpt}")
    else:
        raise ValueError(f"unknown phase {phase!r}")

    # Basic smoke test to check everything is working
    if smoke:
        batch = min(batch, 8)
        epochs = 2
        num_workers = 2
    else:
        num_workers = 4

    # Check and load classes
    train_ld, val_ld, n_classes = get_loaders(input_res, batch, data_root, num_workers)
    assert n_classes == 10, f"expected 10 imagenette classes at {data_root}, got {n_classes}"

    model = ResizerBackbone(resizer, num_classes=n_classes).to(device)

    if prev_ckpt is not None:
        state = torch.load(prev_ckpt, map_location=device, weights_only=True)
        model.backbone.load_state_dict(state["backbone"])
        print(f"[phase B seed {seed}] loaded backbone from {prev_ckpt.name}")

    # Optimizer
    optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    # Exponential decay 0.94 every 2 epochs => gamma^floor(epoch/2)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda e: 0.94 ** (e // 2))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")

    epoch_logs = []
    tag = f"[phase {phase} seed {seed}{' SMOKE' if smoke else ''}]"

    # MLflow Setup
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("talebi_reproduce")
    run_name = f"phase{phase}_seed{seed}{'_smoke' if smoke else ''}"
    
    mlflow.start_run(run_name=run_name)
    try:
        mlflow.log_params({
            "phase": phase, "seed": seed, "epochs": epochs, "smoke": smoke,
            "input_res": input_res, "target_res": target_res, "batch_size": batch,
            "model_type": "TalebiPaperResizer" if resizer else "Bilinear"
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
                scaler.step(optim)
                scaler.update()

                loss_sum += loss.item() * labels.size(0)
                c1 += logits.argmax(1).eq(labels).sum().item()
                total += labels.size(0)

            sched.step()
            train_top1 = 100.0 * c1 / total
            train_loss = loss_sum / total

            val = evaluate(model, val_ld, device)
            dt = time.time() - t0

            log = {"epoch": epoch, "lr": optim.param_groups[0]["lr"],
                   "train_loss": train_loss, "train_top1": train_top1,
                   "val_top1": val["top1"], "val_top5": val["top5"],
                   "epoch_time_s": dt}
            epoch_logs.append(log)

            print(f"{tag} ep {epoch:03d}  lr={log['lr']:.4f}  "
                  f"train_loss={train_loss:.4f}  train_top1={train_top1:.2f}  "
                  f"val_top1={val['top1']:.2f}  val_top5={val['top5']:.2f}  {dt:.1f}s", flush=True)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_top1": train_top1,
                "val_top1": val["top1"],
                "val_top5": val["top5"],
                "lr": optim.param_groups[0]["lr"]
            }, step=epoch)

            if not smoke:
                RESULT_DIR.mkdir(parents=True, exist_ok=True)
                out = RESULT_DIR / f"phase{phase}_seed{seed}.json"
                plot_dir = RESULT_DIR / "plots"

                result = {
                    "phase": phase, "seed": seed, "epochs": epochs, "smoke": smoke,
                    "input_res": input_res, "target_res": target_res, "batch_size": batch,
                    "final_val_top1": val["top1"], "final_val_top5": val["top5"],
                    "cm": val["cm"],
                    "epoch_logs": epoch_logs,
                }

                out.write_text(json.dumps(result, indent=2))
                save_classification_curves(epoch_logs, plot_dir, phase_tag=f"phase{phase}_seed{seed}")
                save_confusion_matrix(val["cm"], plot_dir, phase_tag=f"phase{phase}_seed{seed}")
                mlflow.log_artifact(str(out), artifact_path="results")
                mlflow.log_artifact(str(plot_dir / f"training_curves_phase{phase}_seed{seed}.png"), artifact_path="plots")
                mlflow.log_artifact(str(plot_dir / f"confusion_matrix_phase{phase}_seed{seed}.png"), artifact_path="plots")

        # Phase A saves backbone for Phase B init (both smoke and real, distinct paths)
        if phase == "A":
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            p = _ckpt_path(seed, smoke=smoke)
            torch.save({"backbone": model.backbone.state_dict(), "seed": seed, "epochs": epochs, "smoke": smoke}, p)
            print(f"{tag} saved backbone -> {p.name}")

        if smoke:
            result = {
                "phase": phase, "seed": seed, "epochs": epochs, "smoke": smoke,
                "input_res": input_res, "target_res": target_res, "batch_size": batch,
                "final_val_top1": val["top1"], "final_val_top5": val["top5"],
                "cm": val["cm"],
                "epoch_logs": epoch_logs,
            }

    finally:
        mlflow.end_run()

    return result