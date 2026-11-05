import math
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import mlflow

from src.evaluation.heatmap import make_gt_heatmap
from src.evaluation.centernet_targets import make_centernet_targets
from src.evaluation.node21_detection_metrics import (
    compute_box_detection_metrics, sensitivity_at_fp)
from src.losses.centernet_det import centernet_detection_loss
from src.evaluation.fcos_targets import make_fcos_targets
from src.losses.fcos import fcos_loss
from src.data.node21 import MAX_OBJECTS
from src.evaluation.detection_metrics import compute_detection_metrics_v4, _nms_radius_for
from src.losses.auxiliary import aux_lung_supervised_loss
from src.diagnostics.runner import run_diagnostics, diagnose_gradient_flow

# The false-positive budget this project reports at, and the one the `val_sens_matched`
# checkpoint criterion is judged on. NODE21 itself reports here.
MATCHED_FP_BUDGET = 0.25


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    variant: str,
    epoch: int,
    sigma_min: float,
    loss_fn: nn.Module,
    heatmap_size: int,
    device: torch.device,
    accum_steps: int = 1,
    aux_lambda: float = 0.1,
    input_size: int = 2048,   # coord scale in targets["center_2048"]; MUST match the loader
) -> Tuple[float, Optional[Dict[str, Any]]]:
    model.train()

    # model.train() re-enables BatchNorm updates everywhere, including a backbone the
    # caller has frozen. Put it back into eval if its parameters are not trainable.
    backbone_module = getattr(getattr(model, "backbone_fpn", None), "features", None)
    if backbone_module is not None and not any(p.requires_grad for p in backbone_module.parameters()):
        backbone_module.eval()

    total_loss, finite_batches = 0.0, 0
    grad_stats = None
    grad_flow_checked = False
    pos_mask_logged = False
    aux_stats_logged = False

    # Epoch aggregates for the loss-component line. The previous version printed the FIRST
    # batch only, which is unrepresentative and actively confusing: about 10% of 8-image
    # batches contain no nodule at all (0.75^8 at ~25% positive films), and those printed
    # `giou=0.0000 centerness=0.0000 pos_cells=0` next to a `[pos_mask] sum=87` line taken
    # from a DIFFERENT, later batch - the first one that happened to contain a nodule.
    # Reading those two lines together suggested a supervision bug that was never there.
    part_totals = {"cls": 0.0, "giou": 0.0, "centerness": 0.0}
    part_batches = 0
    positive_cells_total = 0
    empty_positive_batches = 0

    optimizer.zero_grad()
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)

        # Box-detection path: three branches, three-term loss. Kept separate from the
        # heatmap-only path so the older runs stay exactly reproducible.
        if getattr(model, "predicts_boxes", False):
            is_fcos = getattr(model, "is_fcos", False)

            # Both of these MUST come from the batch, not from the run-level scalars.
            # Under pooled NODE21+VinDr training the two datasets pad boxes to different
            # widths (4 vs 16) and may enter at different resolutions (1024 vs 2700).
            # Using the run-level MAX_OBJECTS would silently DROP every box past the
            # fourth on a VinDr film, and using the run-level input_size would scale
            # VinDr boxes by the wrong factor and land them in the wrong grid cells.
            # For any single-source run these are identical to the old values, since the
            # loader always emits square input_size images padded to its own MAX_OBJECTS.
            batch_input_size = images.shape[-1]
            batch_max_objects = targets["boxes"].shape[1]

            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(images)

                if is_fcos:
                    gt = make_fcos_targets(targets, images.shape[0], heatmap_size,
                                           input_size=batch_input_size, device=device,
                                           max_objects=batch_max_objects)
                    loss, loss_parts = fcos_loss(prediction, gt)

                else:
                    gt = make_centernet_targets(targets, images.shape[0], heatmap_size,
                                                input_size=batch_input_size, device=device,
                                                max_objects=batch_max_objects)
                    loss, loss_parts = centernet_detection_loss(prediction, gt)

                loss = loss / accum_steps

            if is_fcos:
                for key in part_totals:
                    part_totals[key] += float(loss_parts[key])

                part_batches += 1
                positive_cells_total += int(loss_parts["n_pos"])

                if int(loss_parts["n_pos"]) == 0: empty_positive_batches += 1

            elif not aux_stats_logged:
                print(f"  [centernet_det] hm={loss_parts['heatmap']:.4f} "
                      f"size={loss_parts['size']:.4f} offset={loss_parts['offset']:.4f} "
                      f"objects={int(gt['mask'].sum())}")
                aux_stats_logged = True

            hm_gt = gt["classification"] if is_fcos else gt["heatmap"]

        else:
          with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hm_pred = model(images)
            # Per-batch for the same reason as the box heads above; identical to
            # input_size for every single-source run.
            hm_gt   = make_gt_heatmap(targets, images.shape[0], heatmap_size, sigma_min=sigma_min, input_size=images.shape[-1], device=device)

            if variant == "talebi_aux":
                  lung_masks = targets["lung_mask"].to(device)
                  loss, aux_info = aux_lung_supervised_loss(
                      hm_pred, hm_gt, model, images, lung_masks, loss_fn, aux_lambda)
                  loss = loss / accum_steps
                  if not aux_stats_logged:
                      print(f"  [talebi_aux] det={aux_info['detection_loss']:.4f} "
                            f"outside={aux_info['aux_outside']:.4f} "
                            f"inside={aux_info['aux_inside']:.4f}")
                      aux_stats_logged = True
            else:
                  loss = loss_fn(hm_pred, hm_gt) / accum_steps


        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (epoch == 1 and batch_idx == 4
                    and variant != "bilinear"
                    and not grad_flow_checked):
                ldm_params = [(n, p) for n, p in model.ldm.named_parameters() if p.requires_grad]
                if ldm_params:
                    has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 for _, p in ldm_params)
                    tag = "OK" if has_grad else "WARNING"
                    verb = "flows through" if has_grad else "NOT FLOWING to"
                    print(f"  [{tag}] Gradient {verb} LDM ({len(ldm_params)} params)")
                    for name, p in ldm_params:
                        grad_max = p.grad.abs().max().item() if p.grad is not None else 0.0
                        print(f"    LDM param {name}: grad_max={grad_max:.8f}")
                grad_flow_checked = True

            # Snapshot LDM gradient stats on the final optimizer step of the epoch.
            # MUST run before optimizer.zero_grad() below. This previously sat after the
            # zeroing, which recorded grad_max=0.0 for every epoch of every logged run.
            if (batch_idx + 1) == len(loader) and variant != "bilinear":
                grad_stats = diagnose_gradient_flow(model, variant)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if not torch.isfinite(loss):
            continue
        total_loss += loss.item() * accum_steps
        finite_batches += 1

    # One line per epoch, averaged over every batch rather than sampled from the first.
    # `empty` is the share of batches with no positive cell at all - a property of the
    # 25%-positive sampling, not a fault, but worth seeing because those batches contribute
    # classification loss only.
    if part_batches:
        print(f"  [fcos epoch mean] cls={part_totals['cls'] / part_batches:.4f} "
              f"giou={part_totals['giou'] / part_batches:.4f} "
              f"centerness={part_totals['centerness'] / part_batches:.4f} | "
              f"pos_cells/batch={positive_cells_total / part_batches:.1f} "
              f"empty={empty_positive_batches}/{part_batches}")

    if finite_batches == 0:
        return float('nan'), grad_stats
    return total_loss / finite_batches, grad_stats

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    sigma_min: float,
    loss_fn: nn.Module,
    heatmap_size: int,
    device: torch.device,
    input_size: int = 2048,
) -> float:
    model.eval()
    total_loss, finite_batches = 0.0, 0

    for images, targets in loader:
        images = images.to(device)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if getattr(model, "predicts_boxes", False):
                prediction = model(images)

                if getattr(model, "is_fcos", False):
                    gt = make_fcos_targets(targets, images.shape[0], heatmap_size,
                                           input_size=input_size, device=device,
                                           max_objects=MAX_OBJECTS)
                    loss, _ = fcos_loss(prediction, gt)

                else:
                    gt = make_centernet_targets(targets, images.shape[0], heatmap_size,
                                                input_size=input_size, device=device,
                                                max_objects=MAX_OBJECTS)
                    loss, _ = centernet_detection_loss(prediction, gt)

            else:
                hm_pred = model(images)
                hm_gt   = make_gt_heatmap(targets, images.shape[0], heatmap_size, sigma_min=sigma_min, input_size=input_size, device=device)
                loss = loss_fn(hm_pred, hm_gt)

        if torch.isfinite(loss):
            total_loss += loss.item()
            finite_batches += 1

    return total_loss / finite_batches if finite_batches else float('nan')

def train_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    variant: str,
    epochs: int,
    patience: int,
    device: torch.device,
    grad_accum: int,
    aux_lambda: float,
    initial_params: Dict[str, torch.Tensor],
    output_dir: Path,
    loss_fn: nn.Module,
    effective_sigma_min: float,
    input_size: int = 2048,
    select_on: str = "val_sens",
    freeze_backbone_epochs: int = 0,
    lung_bbox_512: Tuple[int, int, int, int] = (128, 384, 76, 230), # example coords
    mediastinum_bbox_512: Tuple[int, int, int, int] = (128, 384, 230, 281), # example coords
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    heatmap_size = model.heatmap_size
    history = {
        "train_loss": [], "val_loss": [], "sensitivity": [], "fp_per_image": [], "lr": [],
        "best_val_sensitivity": 0.0, "best_val_sensitivity_epoch": 0,
        "best_val_fp": float("inf"),
        "best_val_loss": float("inf"), "best_val_loss_epoch": 0,
        "sens_matched": [],
        "best_val_sens_matched": 0.0, "best_val_sens_matched_epoch": 0,
        "select_on": select_on,
        "peak_sensitivity": 0.0, "peak_sensitivity_epoch": 0,
    }
    diagnostics_history = []
    patience_counter    = 0
    nan_recoveries      = 0

    # Backbone freeze / warmup. The detection head starts randomly initialised, so its
    # first gradients are noise - and with nothing holding the backbone still, that noise
    # is written straight into ImageNet features worth ~0.40 sensitivity on this task
    # (pretrained resnet50 0.578 vs scratch 0.177). Freezing for the first few epochs
    # lets the head reach a sane state before the backbone is allowed to move.
    backbone_module = getattr(getattr(model, "backbone_fpn", None), "features", None)

    def set_backbone_trainable(trainable):
        if backbone_module is None: return

        for parameter in backbone_module.parameters():
            parameter.requires_grad = trainable

        # eval() during the freeze so BatchNorm running statistics do not drift toward
        # the new domain while the weights are held fixed - a frozen backbone whose BN
        # stats keep moving is not actually frozen.
        backbone_module.train(trainable)

    if freeze_backbone_epochs > 0:
        set_backbone_trainable(False)
        frozen_count = sum(p.numel() for p in backbone_module.parameters()) if backbone_module else 0
        print(f"  [freeze] backbone frozen for {freeze_backbone_epochs} epochs ({frozen_count:,} params)")

    for epoch in range(1, epochs + 1):
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            set_backbone_trainable(True)
            print(f"  [freeze] backbone unfrozen at epoch {epoch}, lr {optimizer.param_groups[0]['lr']:.2e}")

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        if epoch == 1:
            nms_radius = _nms_radius_for(heatmap_size)
            print(f"  [config] heatmap_size={heatmap_size} sigma_min={effective_sigma_min:.2f} "
                  f"scale_factor={input_size // heatmap_size} nms_radius={nms_radius}")

        t_loss, grad_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            variant=variant,
            epoch=epoch,
            sigma_min=effective_sigma_min,
            loss_fn=loss_fn,
            heatmap_size=heatmap_size,
            device=device,
            accum_steps=grad_accum,
            aux_lambda=aux_lambda,
            input_size=input_size,
        )

        if not math.isfinite(t_loss):
            best_ckpt = output_dir / "best_model.pth"
            if not best_ckpt.exists() or nan_recoveries >= 3:
                print(f"[NaN guard] Epoch {epoch}: halting (loss={t_loss}, recoveries={nan_recoveries})")
                break
            nan_recoveries += 1
            for g in optimizer.param_groups:
                g["lr"] *= 0.5
            new_lr = optimizer.param_groups[0]["lr"]
            print(f"[NaN guard] Epoch {epoch}: loss={t_loss}, reducing lr to {new_lr:.2e} "
                  f"(attempt {nan_recoveries}/3)")
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
            scaler = GradScaler(device=device.type, enabled=False)
            continue
        nan_recoveries = 0

        v_loss      = validate(model, val_loader, effective_sigma_min, loss_fn, heatmap_size, device, input_size=input_size)
        if getattr(model, "predicts_boxes", False):
            val_metrics = compute_box_detection_metrics(model, val_loader, device=device, input_size=input_size)
            sens   = val_metrics.get("sensitivity", 0.0)
            fp_img = val_metrics.get("fp_per_image", 0.0)

            # Sensitivity read at the budget this project REPORTS at. The curve is computed
            # every epoch already and was previously discarded; `sens` above is measured at
            # a fixed score_threshold of 0.3, which constrains false positives not at all.
            sens_matched = sensitivity_at_fp(val_metrics.get("froc"), MATCHED_FP_BUDGET)

        else:
            val_metrics = compute_detection_metrics_v4(model, val_loader, device=device, input_size=input_size)
            sens        = val_metrics.get("all", {}).get("sensitivity", 0.0)
            fp_img      = val_metrics.get("all", {}).get("fp_per_image", 0.0)
            sens_matched = 0.0            # heatmap-only path emits no FROC curve

        # Track both criteria every epoch for reporting, independently of which one
        # actually drives checkpointing below.
        if sens > history["best_val_sensitivity"]:
            history["best_val_sensitivity"]       = sens
            history["best_val_sensitivity_epoch"] = epoch
            history["best_val_fp"]                = fp_img

        if v_loss < history["best_val_loss"]:
            history["best_val_loss"]       = v_loss
            history["best_val_loss_epoch"] = epoch

        if sens_matched > history["best_val_sens_matched"]:
            history["best_val_sens_matched"]       = sens_matched
            history["best_val_sens_matched_epoch"] = epoch

        # Checkpoint selection. "val_sens" is the historical default, kept so earlier runs
        # stay reproducible. "val_loss" is the smoother signal: val sensitivity swings by
        # ~0.15 between adjacent epochs on a 939-image val set, so picking its argmax
        # selects a lucky epoch as much as a good model, and the higher the val peak the
        # worse the observed test transfer.
        improved = ""

        if select_on == "val_loss":
            checkpoint_improved = history["best_val_loss_epoch"] == epoch

        elif select_on == "val_sens_matched":
            # Selection agrees with reporting. "val_sens" maximises sensitivity at a fixed
            # score threshold with NO false-positive constraint, so it saves the most liberal
            # epoch the run produced - the worst val loss and highest FP of a run can be the
            # saved checkpoint. This criterion instead asks: at the operating point we quote,
            # which epoch detects most?
            checkpoint_improved = history["best_val_sens_matched_epoch"] == epoch

        else:
            checkpoint_improved = history["best_val_sensitivity_epoch"] == epoch

            # Tie-break: equal sensitivity but fewer false positives is still an improvement
            if not checkpoint_improved and sens == history["best_val_sensitivity"] and fp_img < history["best_val_fp"]:
                history["best_val_fp"] = fp_img
                checkpoint_improved    = True
                improved               = " *(fp)"

        if checkpoint_improved:
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            improved = improved or " *"

        else:
            patience_counter += 1

        if sens > history["peak_sensitivity"]:
            history["peak_sensitivity"]       = sens
            history["peak_sensitivity_epoch"] = epoch

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["sensitivity"].append(sens)
        history["fp_per_image"].append(fp_img)
        history["sens_matched"].append(sens_matched)

        # Sens@0.25 is the number this project actually reports. Sens/FP beside it are at a
        # fixed score threshold of 0.3, which sits far right on the FCOS curve - a high FP
        # there is the threshold, not the model.
        print(f"Ep {epoch:02d}/{epochs} | Train: {t_loss:.4f} | "
              f"Val: {v_loss:.4f} | Sens@{MATCHED_FP_BUDGET}FP: {sens_matched:.3f} | "
              f"Sens@0.3: {sens:.3f} | FP: {fp_img:.2f} | "
              f"LR: {current_lr:.1e}{improved}")

        # Assuming mlflow tracker context is active in outer script
        if mlflow.active_run():
            mlflow.log_metrics({
                "train_loss":       t_loss,
                "val_loss":         v_loss,
                "val_sensitivity":  sens,
                "val_sens_matched": sens_matched,
                "val_fp_per_image": fp_img,
                "lr":               current_lr,
            }, step=epoch)

        if scheduler is not None:
            scheduler.step()

        if (epoch % 3 == 0 or epoch == 1) and variant != "bilinear":
            diag = run_diagnostics(model, val_loader, variant, epoch, initial_params, device, lung_bbox_512, mediastinum_bbox_512)
            if grad_stats and grad_stats.get("has_gradients"):
                diag["grad_flow"] = grad_stats
                drift_l2 = diag.get("param_drift", {}).get("total_l2_drift", 0)
                print(f"  [grads] mean={grad_stats.get('mean_grad_norm', 0):.4f} "
                      f"max={grad_stats.get('max_grad_norm', 0):.4f} | "
                      f"[drift] L2={drift_l2:.3f}")
            diagnostics_history.append(diag)

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} ({patience} epochs without improvement)")
            break

    return history, diagnostics_history
