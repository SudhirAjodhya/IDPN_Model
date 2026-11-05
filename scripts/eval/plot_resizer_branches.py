"""
Decompose the learned resizer into its branches, at both of its addition points.

The Talebi resizer sums twice, and the two sums mean different things:

  OUTER   output = bilinear(x) + output_conv(f)
          A parameter-free geometric skip plus the learned residual. This is the sum that
          makes the arm "at least bilinear at initialisation", because output_conv starts
          near zero.

  INNER   f = post(res_blocks(f_skip)) + f_skip
          f_skip is the input_conv features resampled to the target; the other term is what
          the residual blocks add on top of them. This sum sits INSIDE the residual branch,
          so neither of its terms is an image - both are 16-channel feature maps, shown
          here as the L2 norm across channels.

WHY THE INNER SPLIT IS WORTH SEEING. The outer residual is what reaches the backbone, and
it is anatomical rather than lesion-specific. The inner split says where that comes from:
whether the structure originates in the entry convolutions, which see the image at FULL
input resolution before any resampling, or is built by the residual blocks afterwards at
target resolution.

Feature maps use a sequential colour map because they are magnitudes with no meaningful
zero; the outer residual uses a diverging map centred at zero because it is signed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT_DIR = Path("outputs/consolidation")


def find_run(root, variant, target, seed=42):
    for path in (root / "outputs").rglob("metrics.json"):
        try:
            metrics = json.load(open(path))
        except Exception:
            continue

        if (metrics.get("run_group") in ("v15_selection_and_scale", "v19_focal_norm_fix")
                and metrics.get("variant") == variant
                and metrics.get("target_size") == target
                and metrics.get("seed") == seed
                and metrics.get("input_size") == 1024
                and metrics.get("select_on") == "val_sens_matched"):
            return path.parent

    return None


@torch.no_grad()
def decompose(resizer, image):
    """
    Re-run the resizer's forward, keeping every intermediate.

    Mirrors TalebiPaperResizer.forward exactly. Written out rather than hooked so the
    correspondence to the published architecture stays visible and a change there fails
    loudly here.

    Returns:
        dict of named intermediates, each (H, W) numpy
    """
    size = resizer.target_size

    # OUTER branch one: the geometric skip, parameter-free
    skip = F.interpolate(image, size=(size, size), mode=resizer.skip_mode,
                         align_corners=False, antialias=resizer.skip_antialias)

    # Entry convolutions run at FULL input resolution, then resample to target
    features = resizer.input_conv(image)
    features = F.interpolate(features, size=(size, size), mode="bilinear", align_corners=False)

    inner_skip = features                                   # (B, 16, S, S)
    blocks = resizer.post(resizer.res_blocks(features))     # (B, 16, S, S)
    inner_sum = blocks + inner_skip

    # OUTER branch two: the learned residual
    residual = resizer.output_conv(inner_sum)

    norm = lambda t: t.float().square().sum(dim=1).sqrt()[0].cpu().numpy()
    plain = lambda t: t.float()[0, 0].cpu().numpy()

    return {
        "skip": plain(skip),
        "residual": plain(residual),
        "output": plain(skip + residual),
        "inner_skip": norm(inner_skip),
        "inner_blocks": norm(blocks),
        "inner_sum": norm(inner_sum),
    }


def main():
    global OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--film", default="n1076")
    parser.add_argument("--target", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    arguments = parser.parse_args()

    OUT_DIR = Path(arguments.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root))

    from src.data.splits import build_three_way_split
    from scripts.eval.compare_resizer_outputs import load_model
    import scripts.train as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = find_run(root, "node21_talebi_zeroinit", arguments.target, arguments.seed)

    if run is None:
        raise SystemExit("no learned run at target {}".format(arguments.target))

    model, _ = load_model(run, device)

    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=T.JSRT_CSV_PATH, jsrt_image_dir=T.JSRT_IMAGE_DIR,
        node21_csv_path=T.NODE21_CSV_PATH, node21_image_dir=T.NODE21_IMAGE_DIR,
        datasets="node21", input_size=1024, seed=42, test_seed=0, val_frac=0.20,
        group_by_image=True)

    dataset = test_loaders["node21"]
    inner = getattr(dataset, "dataset", dataset)

    index = next((i for i, (name, boxes) in enumerate(inner.records)
                  if str(name).rsplit(".", 1)[0] == arguments.film and len(boxes)), None)

    if index is None:
        raise SystemExit("film {} not found with a nodule".format(arguments.film))

    image, target = dataset[index]
    parts = decompose(model.ldm, image.unsqueeze(0).to(device))

    box = target["boxes"][0].tolist()
    scale = arguments.target / 1024.0
    cx, cy = box[0] * scale, box[1] * scale

    figure, axes = plt.subplots(2, 3, figsize=(13.2, 9.2))

    outer = [
        ("skip", "Bilinear skip\n$\\mathrm{bilinear}(x)$", "gray", False),
        ("residual", "Learned residual\n$\\mathrm{output\\_conv}(f)$", "RdBu_r", True),
        ("output", "Sum, what the backbone receives", "gray", False),
    ]

    innerp = [
        ("inner_skip", "Entry features\n$\\mathrm{interp}(\\mathrm{input\\_conv}(x))$", "magma", False),
        ("inner_blocks", "Residual blocks\n$\\mathrm{post}(\\mathrm{res\\_blocks}(\\cdot))$", "magma", False),
        ("inner_sum", "Inner sum $f$", "magma", False),
    ]

    for row, group in enumerate((outer, innerp)):
        for column, (key, title, cmap, diverging) in enumerate(group):
            axis = axes[row][column]
            data = parts[key]

            if diverging:
                limit = np.percentile(np.abs(data), 99.5)
                axis.imshow(data, cmap=cmap, vmin=-limit, vmax=limit)
            else:
                axis.imshow(data, cmap=cmap)

            axis.set_title(title, fontsize=10)
            axis.set_xticks([]); axis.set_yticks([])

            # Mark the nodule so the branches can be compared at the lesion
            axis.plot(cx, cy, marker="o", ms=13, mfc="none", mec="#39FF14", mew=1.6)

    axes[0][0].set_ylabel("outer sum\n(image domain)", fontsize=10)
    axes[1][0].set_ylabel("inner sum\n(16-channel features, L2)", fontsize=10)

    figure.suptitle("Resizer branches at both addition points: film {}, target {} px, seed {}"
                    .format(arguments.film, arguments.target, arguments.seed), fontsize=12)
    figure.text(0.5, 0.015,
                "Top: output = bilinear skip + learned residual. Bottom: inside the residual "
                "branch, f = residual-block output + entry features. Feature panels are the "
                "L2 norm over 16 channels. Green circle marks the annotated nodule.",
                ha="center", fontsize=8.5, color="dimgray")
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))

    path = OUT_DIR / "fig15_resizer_branches_{}_t{}.png".format(arguments.film, arguments.target)
    figure.savefig(path, dpi=190)
    print("wrote {}".format(path))

    print("\n  {:<16} {:>12} {:>12}".format("branch", "mean |.|", "std"))

    for key in ("skip", "residual", "inner_skip", "inner_blocks", "inner_sum"):
        print("  {:<16} {:>12.4f} {:>12.4f}".format(
            key, float(np.abs(parts[key]).mean()), float(parts[key].std())))

    ratio = np.abs(parts["inner_blocks"]).mean() / np.abs(parts["inner_skip"]).mean()
    print("\n  residual blocks contribute {:.2f}x the entry features, by mean magnitude".format(ratio))


if __name__ == "__main__":
    main()
