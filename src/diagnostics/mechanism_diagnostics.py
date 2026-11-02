import torch
import torch.nn.functional as F

def diagnose_alpha(model, image_batch, lung_bbox, mediastinum_bbox):
    """
    For Variant alpha: measure |residual| in lung region vs mediastinum region.
    A ratio meaningfully != 1.0 means the residual is content-aware.
    Also reports absolute magnitudes - the ratio alone is misleading when
    both regions are tiny.
    lung_bbox, mediastinum_bbox: (y0, y1, x0, x1) in 512-space
    """
    _ = model(image_batch)  # forward pass to cache residual
    residual = model.ldm.get_last_residual()  # (B, 1, 512, 512)
    if residual is None:
        return {}

    y0, y1, x0, x1 = lung_bbox
    lung_mag = residual[:, :, y0:y1, x0:x1].abs().mean().item()

    my0, my1, mx0, mx1 = mediastinum_bbox
    med_mag = residual[:, :, my0:my1, mx0:mx1].abs().mean().item()

    # Global residual stats
    global_mag = residual.abs().mean().item()
    global_max = residual.abs().max().item()
    global_std = residual.std().item()

    # What fraction of the bilinear signal is the residual?
    base = F.interpolate(image_batch, size=(512, 512), mode='bilinear', align_corners=False)
    residual_to_signal = global_mag / (base.abs().mean().item() + 1e-8)

    ratio = lung_mag / (med_mag + 1e-8)
    return {
        "lung_residual_mag": lung_mag,
        "mediastinum_residual_mag": med_mag,
        "ratio": ratio,
        "global_mean_abs": global_mag,
        "global_max_abs": global_max,
        "global_std": global_std,
        "residual_to_signal_ratio": residual_to_signal,
    }


def diagnose_gamma(model, image_batch, lung_bbox, mediastinum_bbox):
    """
    For Variant gamma: measure the entropy of the per-block weight distribution.
    Low entropy in lung region + high entropy in mediastinum means
    the operator has learned to selectively focus on lung pixels.
    """
    _ = model(image_batch)
    weights = model.ldm.get_last_weights()  # (B, 1, 512, 4, 512, 4)
    if weights is None:
        return {}

    weights_flat = weights.reshape(weights.shape[0], 1, 512, 512, 16)
    log_weights = torch.log(weights_flat + 1e-8)
    entropy_per_block = -(weights_flat * log_weights).sum(dim=-1)  # (B, 1, 512, 512)

    y0, y1, x0, x1 = lung_bbox
    lung_entropy = entropy_per_block[:, :, y0:y1, x0:x1].mean().item()
    
    my0, my1, mx0, mx1 = mediastinum_bbox
    med_entropy = entropy_per_block[:, :, my0:my1, mx0:mx1].mean().item()

    # Raw weight stats - entropy alone doesn't say HOW uniform the weights are.
    # Uniform weight = 1/16 = 0.0625; "peaked" block = max weight > 2/16 = 0.125.
    max_weight = weights_flat.max(dim=-1).values.mean().item()
    min_weight = weights_flat.min(dim=-1).values.mean().item()
    weight_std = weights_flat.std(dim=-1).mean().item()  # per-block std
    peaked_frac = (weights_flat.max(dim=-1).values > 0.125).float().mean().item()

    # Saliency map stats
    saliency = model.ldm.get_last_saliency()
    sal_stats = {}
    if saliency is not None:
        sal_stats = {
            "saliency_mean": saliency.mean().item(),
            "saliency_std": saliency.std().item(),
            "saliency_min": saliency.min().item(),
            "saliency_max": saliency.max().item(),
            "saliency_range": (saliency.max() - saliency.min()).item(),
        }

    # Uniform entropy for 16 elements: log(16) approx 2.773
    return {
        "lung_entropy": lung_entropy,
        "mediastinum_entropy": med_entropy,
        "uniform_reference": 2.7726,
        "lung_is_peakier": lung_entropy < med_entropy,
        "mean_max_weight": max_weight,
        "mean_min_weight": min_weight,
        "mean_per_block_std": weight_std,
        "frac_peaked_blocks": peaked_frac,
        **sal_stats,
    }


def diagnose_gamma_innet(model, image_batch):
    """
    Same idea as diagnose_gamma but for feature-map input.
    We don't have anatomical bboxes at feature-map resolution, so instead
    measure global entropy stats.
    """
    _ = model(image_batch)
    # LDM is wrapped inside backbone_fpn
    if not hasattr(model.backbone_fpn, 'get_last_weights'):
        return {}
    weights = model.backbone_fpn.get_last_weights()  # (B, 1, H, 2, W, 2)
    if weights is None:
        return {}

    B, _, H, _, W, _ = weights.shape
    weights_flat = weights.reshape(B, 1, H, W, 4)
    log_weights = torch.log(weights_flat + 1e-8)
    entropy = -(weights_flat * log_weights).sum(dim=-1)  # (B, 1, H, W)

    uniform_ref = torch.log(torch.tensor(4.0)).item()  # log(4) approx 1.386

    return {
        "mean_entropy": entropy.mean().item(),
        "std_entropy": entropy.std().item(),
        "uniform_reference": uniform_ref,
        "frac_low_entropy": (entropy < 0.5 * uniform_ref).float().mean().item(),
    }


def diagnose_bilinear_mse(model, image_batch):
    """
    Compute MSE between the mechanism's 512x512 output and bilinear's 512x512 output.
    If MSE approx 0, the mechanism is functionally identical to bilinear.
    For alpha/gamma only - gamma-in-net's LDM lives inside the backbone (see
    diagnose_innet_bilinear_mse).
    """
    model.eval()
    with torch.no_grad():
        mech_output = model.ldm(image_batch)  # (B, 1, 512, 512)
        bilinear_output = F.interpolate(image_batch, size=(512, 512),
                                        mode='bilinear', align_corners=False)

        mse = F.mse_loss(mech_output, bilinear_output).item()
        rmse = mse ** 0.5
        max_diff = (mech_output - bilinear_output).abs().max().item()
        mean_diff = (mech_output - bilinear_output).abs().mean().item()

    return {
        "mse_vs_bilinear": mse,
        "rmse_vs_bilinear": rmse,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
    }


def diagnose_innet_bilinear_mse(model, image_batch):
    """
    For gamma-in-net: compare InNetworkWeightedPooling output vs avg_pool2d on the
    same input, captured via a forward hook on the module.
    """
    ldm = getattr(model.backbone_fpn, 'ldm', None)
    if ldm is None or not hasattr(ldm, 'weighted_pool'):
        return {}

    captured = {}

    def hook_fn(module, input, output):
        captured['input'] = input[0].detach()
        captured['output'] = output.detach()

    handle = ldm.weighted_pool.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        _ = model(image_batch)
    handle.remove()

    if 'input' not in captured:
        return {}

    inp = captured['input']            # (B, C, H, W)
    mech_out = captured['output']      # (B, C, H//2, W//2)
    avgpool_ref = F.avg_pool2d(inp, kernel_size=2)  # simple 2x2 average

    mse = F.mse_loss(mech_out, avgpool_ref).item()

    return {
        "innet_mse_vs_avgpool": mse,
        "innet_rmse_vs_avgpool": mse ** 0.5,
        "innet_max_diff": (mech_out - avgpool_ref).abs().max().item(),
    }


def diagnose_lip(model, image_batch, lung_bbox, mediastinum_bbox):
    """
    For Variant epsilon (LIP): measure whether importance weights favour lung over mediastinum.
    importance_ratio = mean(exp(logit_lung)) / mean(exp(logit_med)).
    Should be >1.5 if content-aware in the right direction.
    """
    ldm = model.ldm
    with torch.no_grad():
        logit1 = ldm.logit1(image_batch)
        logit1_amp = ldm.amp * torch.sigmoid(logit1)
        weight1 = torch.exp(logit1_amp)

        y0, y1, x0, x1 = lung_bbox
        lung_weight = weight1[:, :, y0:y1, x0:x1].mean().item()
        my0, my1, mx0, mx1 = mediastinum_bbox
        med_weight = weight1[:, :, my0:my1, mx0:mx1].mean().item()

    return {
        "lung_importance":  lung_weight,
        "med_importance":   med_weight,
        "importance_ratio": lung_weight / (med_weight + 1e-8),
        "logit_std":        logit1.std().item(),
        "logit_max":        logit1.max().item(),
        "logit_min":        logit1.min().item(),
    }


def diagnose_talebi(model, image_batch, lung_bbox, mediastinum_bbox, target_size=None):
    """
    Residual magnitude, spatial distribution (lung/mediastinum), MSE vs bilinear.
    lung_bbox, mediastinum_bbox: (y0, y1, x0, x1) in target_size-space.
    target_size defaults to whatever the resizer actually emits - avoids the
    hardcoded 512 crashing when target=224 was set from CLI.
    """
    ldm = model.ldm
    with torch.no_grad():
        output   = ldm(image_batch)
        if target_size is None:
            target_size = output.shape[-1]
        bilinear = F.interpolate(image_batch, size=(target_size, target_size),
                                  mode="bilinear", align_corners=False)
        residual = output - bilinear

    y0, y1, x0, x1 = lung_bbox
    lung_resid = residual[:, :, y0:y1, x0:x1].abs().mean().item()
    y0, y1, x0, x1 = mediastinum_bbox
    med_resid  = residual[:, :, y0:y1, x0:x1].abs().mean().item()

    return {
        "residual_mean_abs":  residual.abs().mean().item(),
        "residual_max":       residual.abs().max().item(),
        "residual_std":       residual.std().item(),
        "lung_residual":      lung_resid,
        "med_residual":       med_resid,
        "lung_to_med_ratio":  lung_resid / (med_resid + 1e-8),
        "mse_vs_bilinear":    residual.pow(2).mean().item(),
    }


def _saliency_params(model, variant):
    """The mechanism's saliency/residual parameters for each variant."""
    if variant == "alpha":
        return list(model.ldm.stem.parameters()) + list(model.ldm.head.parameters()), "alpha residual"
    elif variant == "gamma":
        return (list(model.ldm.saliency_stem.parameters())
                + list(model.ldm.saliency_head.parameters())), "gamma saliency"
    elif variant == "gamma_innet":
        ldm = getattr(model.backbone_fpn, 'ldm', None)
        if ldm is None:
            return None, None
        return list(ldm.weighted_pool.saliency.parameters()), "gamma-innet saliency"
    elif variant == "lip":
        return list(model.ldm.parameters()), "LIP"
    elif variant in ("talebi", "talebi_aux"):
        return list(model.ldm.parameters()), f"{variant} LDM"
    return None, None


def diagnose_gradient_flow(model, variant):
    """
    After loss.backward(), check gradient norms on the mechanism's saliency
    parameters. Call AFTER scaler.scale(loss).backward() but BEFORE
    scaler.step(optimizer). Near-zero norms mean the mechanism cannot learn
    even in principle.
    """
    params, label = _saliency_params(model, variant)
    if params is None:
        return {}

    grad_norms = []
    for p in params:
        if p.grad is not None:
            grad_norms.append(p.grad.data.norm(2).item())

    if not grad_norms:
        return {"label": label, "has_gradients": False}

    return {
        "label": label,
        "has_gradients": True,
        "mean_grad_norm": sum(grad_norms) / len(grad_norms),
        "max_grad_norm": max(grad_norms),
        "min_grad_norm": min(grad_norms),
        "num_params_with_grad": len(grad_norms),
    }


def _saliency_named_params(model, variant):
    """Named parameter dict of the mechanism module, for drift tracking."""
    if variant in ("alpha", "gamma"):
        return dict(model.ldm.named_parameters())
    elif variant == "gamma_innet":
        ldm = getattr(model.backbone_fpn, 'ldm', None)
        if ldm is None:
            return {}
        return dict(ldm.named_parameters())
    elif variant in ("lip", "talebi", "talebi_aux"):
        return dict(model.ldm.named_parameters())
    return {}


def snapshot_saliency_params(model, variant):
    """
    Snapshot of saliency parameter values. Call before training starts.
    Returns {name: tensor_clone}.
    """
    return {n: p.detach().clone() for n, p in _saliency_named_params(model, variant).items()}


def diagnose_param_drift(model, variant, initial_snapshot):
    """
    Compare current saliency params to the initial snapshot (L2 distance).
    If the params haven't moved, the mechanism isn't learning.
    """
    current = _saliency_named_params(model, variant)
    if not current:
        return {}

    total_l2 = 0.0
    total_params = 0
    for name, init_val in initial_snapshot.items():
        if name in current:
            diff = (current[name].detach() - init_val).norm(2).item()
            total_l2 += diff
            total_params += 1

    return {
        "total_l2_drift": total_l2,
        "mean_l2_drift": total_l2 / max(total_params, 1),
        "num_param_groups": total_params,
    }


def diagnose_lip_intra(model, images, active_transitions):
    """
    Verify LIP is learning non-uniform weights inside the ConvNeXt backbone.

    For each active transition, reports logit std and weight peakedness
    (max/mean ratio). Expected: std approx 0 and peakedness approx 1 at epoch 1 (zero-init
    reduces to avg_pool); std>0.1 and peakedness>1.5 by epoch 10+ if learning.
    """
    stats = {}
    model.eval()
    with torch.no_grad():
        x  = model.backbone_fpn.stem(images)
        c1 = model.backbone_fpn.stage1(x)

        if "s1_s2" in active_transitions:
            lm = model.backbone_fpn.down1
            x_n = lm.norm(c1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            logit  = lm.logit(x_n)
            weight = torch.exp(lm.amp * torch.sigmoid(logit))
            stats["s1_s2"] = {
                "logit_std":       logit.std().item(),
                "logit_max":       logit.max().item(),
                "logit_min":       logit.min().item(),
                "weight_peakedness": (weight.max() / weight.mean()).item(),
                "feature_map_size":  tuple(logit.shape[-2:]),
            }

        x  = model.backbone_fpn.down1(c1)
        c2 = model.backbone_fpn.stage2(x)

        if "s2_s3" in active_transitions:
            lm = model.backbone_fpn.down2
            x_n = lm.norm(c2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            logit  = lm.logit(x_n)
            weight = torch.exp(lm.amp * torch.sigmoid(logit))
            stats["s2_s3"] = {
                "logit_std":       logit.std().item(),
                "logit_max":       logit.max().item(),
                "logit_min":       logit.min().item(),
                "weight_peakedness": (weight.max() / weight.mean()).item(),
                "feature_map_size":  tuple(logit.shape[-2:]),
            }

        x  = model.backbone_fpn.down2(c2)
        c3 = model.backbone_fpn.stage3(x)

        if "s3_s4" in active_transitions:
            lm = model.backbone_fpn.down3
            x_n = lm.norm(c3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            logit  = lm.logit(x_n)
            weight = torch.exp(lm.amp * torch.sigmoid(logit))
            stats["s3_s4"] = {
                "logit_std":       logit.std().item(),
                "logit_max":       logit.max().item(),
                "logit_min":       logit.min().item(),
                "weight_peakedness": (weight.max() / weight.mean()).item(),
                "feature_map_size":  tuple(logit.shape[-2:]),
            }

    return stats


def diagnose_delta(model, image_batch):
    """For Variant delta: kernel drift from bilinear init and center-weight sharpness."""
    ldm = model.ldm
    if not hasattr(ldm, 'stage1'):
        return {}

    k1 = ldm.stage1.weight.data[0, 0].cpu()
    k2 = ldm.stage2.weight.data[0, 0].cpu()

    k = k1.shape[0]
    c = k // 2
    bilinear_ref = torch.zeros(k, k)
    for i in range(k):
        for j in range(k):
            bilinear_ref[i, j] = (max(0, 1 - abs(i - c) / (c + 1))
                                  * max(0, 1 - abs(j - c) / (c + 1)))
    bilinear_ref /= bilinear_ref.sum()

    drift1 = (k1 - bilinear_ref).norm(2).item()
    drift2 = (k2 - bilinear_ref).norm(2).item()
    center_weight_1 = k1[c, c].item()
    center_weight_ref = bilinear_ref[c, c].item()

    return {
        "stage1_drift_from_bilinear": drift1,
        "stage2_drift_from_bilinear": drift2,
        "stage1_center_weight": center_weight_1,
        "bilinear_center_weight": center_weight_ref,
        "sharper_than_bilinear": center_weight_1 > center_weight_ref,
        "stage1_kernel_sum": k1.sum().item(),
        "stage2_kernel_sum": k2.sum().item(),
    }