import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.amp import autocast

def save_ldm_comparison(model, test_loaders, output_dir, device, input_size=2048, target_size=512):
    model.eval()
    img_sample, tgt_sample = None, None

    # Prefer JSRT for the sample image (matches historical outputs). Fall back to
    # node21 (or combined) when running Node21-only, so the artifact still gets written.
    sample_loader = test_loaders.get("jsrt") or test_loaders.get("node21") or test_loaders["combined"]
    for images, targets in sample_loader:
        for i in range(images.shape[0]):
            if targets["has_nodule"][i]:
                img_sample = images[i:i+1].to(device)
                tgt_sample = {
                    "has_nodule": bool(targets["has_nodule"][i]),
                    "center_2048": (targets["center_2048"][0][i].item(),
                                    targets["center_2048"][1][i].item()),
                }
                break
        if img_sample is not None:
            break

    if img_sample is None:
        return

    with torch.no_grad():
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hm_pred = model(img_sample)
        ldm_out = model.get_ldm_output(img_sample)

    bilinear_512 = F.interpolate(img_sample, (target_size, target_size),
                                  mode="bilinear", align_corners=False).cpu().squeeze().float()
    variant_512  = ldm_out.cpu().squeeze().float()
    diff         = (variant_512 - bilinear_512).abs()
    pred_hm      = torch.sigmoid(hm_pred).cpu().squeeze().float()

    mse      = diff.pow(2).mean().item()
    max_diff = diff.max().item()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(bilinear_512.numpy(), cmap="gray"); axes[0].set_title("Bilinear 512"); axes[0].axis("off")
    axes[1].imshow(variant_512.numpy(),  cmap="gray"); axes[1].set_title("Variant 512");  axes[1].axis("off")
    im = axes[2].imshow(diff.numpy(), cmap="viridis")
    axes[2].set_title("|Variant - Bilinear|"); axes[2].axis("off")
    axes[2].text(0.02, 0.98, f"MSE={mse:.6f}\nMax={max_diff:.4f}",
                 transform=axes[2].transAxes, va='top', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.8))
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].imshow(pred_hm.numpy(), cmap="hot")
    if tgt_sample["has_nodule"]:
        gx = tgt_sample["center_2048"][0] * (model.heatmap_size / input_size)
        gy = tgt_sample["center_2048"][1] * (model.heatmap_size / input_size)
        axes[3].plot(gx, gy, "g+", markersize=15, markeredgewidth=2)
    axes[3].set_title("Detection Heatmap (+ = GT)"); axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "ldm_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
