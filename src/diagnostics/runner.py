from .mechanism_diagnostics import (
    diagnose_alpha, diagnose_gamma, diagnose_bilinear_mse,
    diagnose_gradient_flow, snapshot_saliency_params, diagnose_param_drift,
    diagnose_delta, diagnose_lip, diagnose_talebi,
)

def run_diagnostics(model, val_loader, variant, epoch, initial_params, device, lung_bbox_512, mediastinum_bbox_512):
    diag_images, _ = next(iter(val_loader))
    diag_images = diag_images.to(device)
    d_stats, mse_stats, drift = {}, {}, {}

    if variant == "alpha":
        d_stats   = diagnose_alpha(model, diag_images, lung_bbox_512, mediastinum_bbox_512)
        mse_stats = diagnose_bilinear_mse(model, diag_images)
        print(f"  [alpha] lung:med={d_stats.get('ratio', -1):.3f} "
              f"resid/signal={d_stats.get('residual_to_signal_ratio', -1):.3f} "
              f"mse_vs_bilinear={mse_stats.get('mse_vs_bilinear', -1):.6f}")

    elif variant == "talebi":
        d_stats   = diagnose_talebi(model, diag_images, lung_bbox_512, mediastinum_bbox_512)
        mse_stats = {"mse_vs_bilinear": d_stats.get("mse_vs_bilinear", -1)}
        ratio     = d_stats.get("lung_to_med_ratio", -1)
        print(f"  [talebi] resid_mean={d_stats.get('residual_mean_abs', -1):.6f} "
              f"lung:med={ratio:.3f} "
              f"({'lung-focused' if ratio > 1 else 'med-focused'}) "
              f"mse_vs_bilinear={mse_stats.get('mse_vs_bilinear', -1):.6f}")

    elif variant == "talebi_aux":
        d_stats   = diagnose_talebi(model, diag_images, lung_bbox_512, mediastinum_bbox_512)
        mse_stats = {"mse_vs_bilinear": d_stats.get("mse_vs_bilinear", -1)}
        ratio     = d_stats.get("lung_to_med_ratio", -1)
        print(f"  [talebi_aux] resid_mean={d_stats.get('residual_mean_abs', -1):.4f} "
              f"lung:med={ratio:.3f} "
              f"({'aux working' if ratio > 1 else 'aux not steering yet'})")

    elif variant == "delta":
        d_stats   = diagnose_delta(model, diag_images)
        mse_stats = diagnose_bilinear_mse(model, diag_images)
        sharper   = "sharper" if d_stats.get("sharper_than_bilinear") else "smoother"
        print(f"  [delta] drift_s1={d_stats.get('stage1_drift_from_bilinear', -1):.3f} "
              f"drift_s2={d_stats.get('stage2_drift_from_bilinear', -1):.3f} "
              f"({sharper}) "
              f"mse_vs_bilinear={mse_stats.get('mse_vs_bilinear', -1):.6f}")

    elif variant == "lip":
        d_stats   = diagnose_lip(model, diag_images, lung_bbox_512, mediastinum_bbox_512)
        mse_stats = diagnose_bilinear_mse(model, diag_images)
        print(f"  [lip] ratio={d_stats.get('importance_ratio', -1):.3f} "
              f"mse_vs_bilinear={mse_stats.get('mse_vs_bilinear', -1):.6f}")

    if initial_params:
        drift = diagnose_param_drift(model, variant, initial_params)

    return {"epoch": epoch, **d_stats, **mse_stats, "param_drift": drift}
