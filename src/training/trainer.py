import math
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import mlflow

from src.evaluation.heatmap import make_gt_heatmap
from src.evaluation.detection_metrics import compute_detection_metrics_v4, _nms_radius_for
from src.losses.auxiliary import aux_lung_supervised_loss
from src.diagnostics.runner import run_diagnostics, diagnose_gradient_flow

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
    total_loss, finite_batches = 0.0, 0
    grad_stats = None
    grad_flow_checked = False
    pos_mask_logged = False
    aux_stats_logged = False

    optimizer.zero_grad()
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)

        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hm_pred = model(images)
            hm_gt   = make_gt_heatmap(targets, images.shape[0], heatmap_size, sigma_min=sigma_min, input_size=input_size, device=device)

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

        if not pos_mask_logged and any(targets["has_nodule"]):
            with torch.no_grad():
                pm = hm_gt.ge(0.99).float()
                print(f"  [pos_mask] sum={pm.sum().item():.0f} "
                      f"(batch has_nodule={targets['has_nodule'].tolist()})")
            pos_mask_logged = True

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (epoch == 1 and batch_idx == 4
                    and variant not in ("bilinear", "blurpool")
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

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if batch_idx == len(loader) - 1 and variant != "bilinear":
            grad_stats = diagnose_gradient_flow(model, variant)

        if not torch.isfinite(loss):
            continue
        total_loss += loss.item() * accum_steps
        finite_batches += 1

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
    lung_bbox_512: Tuple[int, int, int, int] = (128, 384, 76, 230), # example coords
    mediastinum_bbox_512: Tuple[int, int, int, int] = (128, 384, 230, 281), # example coords
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    heatmap_size = model.heatmap_size
    history = {
        "train_loss": [], "val_loss": [], "sensitivity": [], "fp_per_image": [], "lr": [],
        "best_val_sensitivity": 0.0, "best_val_sensitivity_epoch": 0,
        "best_val_fp": float("inf"),
        "peak_sensitivity": 0.0, "peak_sensitivity_epoch": 0,
    }
    diagnostics_history = []
    patience_counter    = 0
    nan_recoveries      = 0

    for epoch in range(1, epochs + 1):
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
        val_metrics = compute_detection_metrics_v4(model, val_loader, device=device, input_size=input_size)
        sens        = val_metrics.get("all", {}).get("sensitivity", 0.0)
        fp_img      = val_metrics.get("all", {}).get("fp_per_image", 0.0)

        improved = ""
        if sens > history["best_val_sensitivity"]:
            history["best_val_sensitivity"]       = sens
            history["best_val_sensitivity_epoch"] = epoch
            history["best_val_fp"]                = fp_img
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            improved = " *"
        elif (sens == history["best_val_sensitivity"]
              and fp_img < history["best_val_fp"]):
            history["best_val_fp"] = fp_img
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            improved = " *(fp)"
        else:
            patience_counter += 1

        if sens > history["peak_sensitivity"]:
            history["peak_sensitivity"]       = sens
            history["peak_sensitivity_epoch"] = epoch

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["sensitivity"].append(sens)
        history["fp_per_image"].append(fp_img)

        print(f"Ep {epoch:02d}/{epochs} | Train: {t_loss:.4f} | "
              f"Val: {v_loss:.4f} | Sens: {sens:.3f} | FP: {fp_img:.2f} | "
              f"LR: {current_lr:.1e}{improved}")

        # Assuming mlflow tracker context is active in outer script
        if mlflow.active_run():
            mlflow.log_metrics({
                "train_loss":       t_loss,
                "val_loss":         v_loss,
                "val_sensitivity":  sens,
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
