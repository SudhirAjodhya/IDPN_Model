"""
What the learned residual does to the image, and how that changes with decimation.

THE CLAIM THIS FIGURE CARRIES. The frequency measurement says the resizer amplifies mid
and high spatial frequencies at high decimation and attenuates them at low decimation,
with the crossover near 2x. That is a table of ratios; it does not show what the residual
looks like, where it sits on the anatomy, or that the same architecture is learning
qualitatively different filters at different output sizes.

LAYOUT. One column per output resolution. Top row: the residual over the whole film,
`learned - bilinear`, on a diverging scale centred at zero, because the quantity is signed
and a sequential ramp would hide the sign. Middle row: the same residual over a crop
around the annotated nodule. Bottom panel: the radial power spectrum of each arm's output,
normalised by the squared contrast gain so the curves are comparable - the resizer also
rescales intensity, and without dividing that out every band is simply shifted up and the
shape is invisible.

Each column uses its own symmetric colour limit. A shared limit would be dominated by the
largest-magnitude target and flatten the rest to grey; the point here is the spatial
STRUCTURE of the residual at each decimation, not its absolute size, which the gain row
reports numerically.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path("outputs/consolidation")

BAND_EDGES = ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0))


def find_run(root, variant, target, seed=42):
    """The seed-42 run for this variant and target, from the corrected run groups."""
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


def radial_spectrum(image, n_bins=40):
    """Mean power per radial frequency bin, DC removed."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image - image.mean()))) ** 2
    size = spectrum.shape[0]
    centre = size // 2

    y, x = np.ogrid[:size, :size]
    radius = np.sqrt((y - centre) ** 2 + (x - centre) ** 2) / centre

    edges = np.linspace(0, 1.0, n_bins + 1)
    centres, values = [], []

    for low, high in zip(edges[:-1], edges[1:]):
        mask = (radius >= low) & (radius < high)

        if mask.any():
            centres.append((low + high) / 2)
            values.append(spectrum[mask].mean())

    return np.array(centres), np.array(values)


def main():
    global OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--film", default="n1076", help="test film stem to illustrate")
    parser.add_argument("--targets", type=int, nargs="*", default=[224, 384, 768, 1024])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    arguments = parser.parse_args()

    OUT_DIR = Path(arguments.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from torch.utils.data import DataLoader

    root = Path(__file__).resolve().parent.parent.parent
    import sys
    sys.path.insert(0, str(root))

    from src.data.splits import build_three_way_split
    from scripts.eval.compare_resizer_outputs import load_model
    import scripts.train as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=T.JSRT_CSV_PATH, jsrt_image_dir=T.JSRT_IMAGE_DIR,
        node21_csv_path=T.NODE21_CSV_PATH, node21_image_dir=T.NODE21_IMAGE_DIR,
        datasets="node21", input_size=1024, seed=42, test_seed=0, val_frac=0.20,
        group_by_image=True)

    dataset = test_loaders["node21"]
    inner = getattr(dataset, "dataset", dataset)

    chosen = None

    for index, (name, boxes) in enumerate(inner.records):
        if str(name).rsplit(".", 1)[0] == arguments.film and len(boxes):
            chosen = index
            break

    if chosen is None:
        raise SystemExit("film {} not found with a nodule in the test split".format(arguments.film))

    image, target = dataset[chosen]
    image = image.unsqueeze(0).to(device)

    box = target["boxes"][0].tolist()          # cx, cy, w, h in 1024-pixel space
    targets = list(arguments.targets)

    figure = plt.figure(figsize=(3.6 * len(targets), 9.6))
    grid = GridSpec(3, len(targets), figure=figure, height_ratios=[1.0, 1.0, 0.85],
                    hspace=0.28, wspace=0.12)

    spectrum_axis = figure.add_subplot(grid[2, :])
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(targets)))

    summary = []

    for column, size in enumerate(targets):
        run_a = find_run(root, "bilinear", size, arguments.seed)
        run_b = find_run(root, "node21_talebi_zeroinit", size, arguments.seed)

        if run_a is None or run_b is None:
            print("skipping target {} - run missing".format(size))
            continue

        model_a, _ = load_model(run_a, device)
        model_b, _ = load_model(run_b, device)

        with torch.no_grad():
            out_a = model_a.ldm(image).float().cpu().numpy()[0, 0]
            out_b = model_b.ldm(image).float().cpu().numpy()[0, 0]

        residual = out_b - out_a
        gain = out_b.std() / out_a.std()

        # Symmetric limit per column: the structure matters here, not the magnitude
        limit = np.percentile(np.abs(residual), 99.5)

        axis = figure.add_subplot(grid[0, column])
        axis.imshow(residual, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_title("{} px   {:.2f}$\\times$ decimation\ncontrast gain {:.2f}$\\times$".format(
            size, 1024 / size, gain), fontsize=10)
        axis.set_xticks([]); axis.set_yticks([])

        if column == 0:
            axis.set_ylabel("residual\n(learned $-$ bilinear)", fontsize=9)

        # Nodule crop, in this target's pixel coordinates
        scale = size / 1024.0
        cx, cy, width, height = (v * scale for v in box)
        half = max(width, height) * 1.8
        x0, x1 = int(max(cx - half, 0)), int(min(cx + half, size))
        y0, y1 = int(max(cy - half, 0)), int(min(cy + half, size))

        crop_axis = figure.add_subplot(grid[1, column])
        crop_axis.imshow(residual[y0:y1, x0:x1], cmap="RdBu_r", vmin=-limit, vmax=limit)
        crop_axis.set_xticks([]); crop_axis.set_yticks([])

        if column == 0:
            crop_axis.set_ylabel("nodule crop", fontsize=9)

        # Spectrum, with the contrast gain divided out so the shapes compare
        freq, power_a = radial_spectrum(out_a)
        _, power_b = radial_spectrum(out_b)
        ratio = (power_b / np.maximum(power_a, 1e-12)) / (gain ** 2)

        spectrum_axis.plot(freq, ratio, color=colours[column], lw=2.0,
                           label="{} px ({:.1f}$\\times$)".format(size, 1024 / size))

        bands = []

        for low, high in BAND_EDGES:
            mask = (freq >= low) & (freq < high)
            bands.append(float(ratio[mask].mean()) if mask.any() else float("nan"))

        summary.append((size, 1024 / size, gain, bands))

        del model_a, model_b
        torch.cuda.empty_cache()

    spectrum_axis.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.6)
    spectrum_axis.text(0.995, 1.0, " no change beyond the contrast gain", va="bottom",
                       ha="right", fontsize=8.5, color="dimgray")
    spectrum_axis.set(xlabel="Radial spatial frequency (fraction of Nyquist)",
                      ylabel="Learned / bilinear power,\nnormalised by gain$^2$",
                      xlim=(0, 1), yscale="log")
    spectrum_axis.legend(frameon=False, fontsize=9, ncol=len(targets), loc="upper left")
    spectrum_axis.grid(alpha=0.2)
    spectrum_axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("The learned residual inverts with decimation (film {}, seed {})".format(
        arguments.film, arguments.seed), fontsize=12)

    path = OUT_DIR / "fig14_residual_across_decimation_{}.png".format(arguments.film)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    print("wrote {}".format(path))

    print("\n  {:<8} {:>8} {:>8} {:>10} {:>10} {:>10}".format(
        "target", "decim", "gain", "low", "mid", "high"))

    for size, decimation, gain, bands in summary:
        print("  {:<8} {:>7.2f}x {:>8.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
            size, decimation, gain, *bands))


if __name__ == "__main__":
    main()
