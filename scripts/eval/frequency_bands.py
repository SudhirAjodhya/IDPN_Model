"""
Does the learned resizer sharpen or blur, and does that change with decimation?

For each output resolution, both arms' resizer outputs are compared on the same NODE21
test films. Power is averaged over three radial frequency bands, as a fraction of
Nyquist: low [0, 0.1), mid [0.1, 0.4), high [0.4, 1.0). The learned resizer also
rescales intensity, which multiplies every band by the squared contrast gain, so each
ratio is divided by gain^2. A value above 1 means the learned output carries more of that
band than a pure contrast stretch of the bilinear output would.

These are the frequency figures quoted in the Discussion. The script prints them and the
matching LaTeX rows.

Usage (from Model/):
    python -m scripts.eval.frequency_bands
    python -m scripts.eval.frequency_bands --films 20 --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval.compare_resizer_outputs import load_model       # noqa: E402
from scripts.eval.plot_resizer_branches import find_run           # noqa: E402
from src.data.splits import build_three_way_split                 # noqa: E402
import scripts.train as T                                         # noqa: E402

BAND_EDGES = ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0))
TARGETS = (224, 256, 384, 512, 768, 1024)
INPUT_SIZE = 1024


def band_power(image):
    """Mean spectral power in each radial band, DC removed. (H, W) -> [low, mid, high]."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image - image.mean()))) ** 2
    side = spectrum.shape[0]
    centre = side // 2

    y, x = np.ogrid[:side, :side]
    radius = np.sqrt((y - centre) ** 2 + (x - centre) ** 2) / centre

    return [float(spectrum[(radius >= low) & (radius < high)].mean()) for low, high in BAND_EDGES]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--films", type=int, default=20, help="first N films of the test split")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The fixed NODE21 test split, in its own order, unaugmented
    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=T.JSRT_CSV_PATH, jsrt_image_dir=T.JSRT_IMAGE_DIR,
        node21_csv_path=T.NODE21_CSV_PATH, node21_image_dir=T.NODE21_IMAGE_DIR,
        datasets="node21", input_size=INPUT_SIZE, seed=42, test_seed=0, val_frac=0.20,
        group_by_image=True)

    films = []

    for batch in DataLoader(test_loaders["node21"], batch_size=1, shuffle=False, num_workers=0):
        if len(films) >= arguments.films: break

        films.append(batch[0] if isinstance(batch, (list, tuple)) else batch["image"])

    print("NODE21, input {}, seed {}, {} test films".format(INPUT_SIZE, arguments.seed, len(films)))
    print("{:<8} {:>8} {:>8} {:>10} {:>10} {:>10}".format(
        "target", "decim", "gain", "low", "mid", "high"))

    rows = []

    for target in TARGETS:
        baseline_run = find_run(ROOT, "bilinear", target, arguments.seed)
        learned_run = find_run(ROOT, "node21_talebi_zeroinit", target, arguments.seed)

        if baseline_run is None or learned_run is None:
            print("{:<8} missing run".format(target))
            continue

        baseline, _ = load_model(baseline_run, device)
        learned, _ = load_model(learned_run, device)

        baseline_bands, learned_bands, gains = [], [], []

        for image in films:
            image = image.to(device)

            with torch.no_grad():
                baseline_out = baseline.ldm(image).float().cpu().numpy()[0, 0]      # (S, S)
                learned_out = learned.ldm(image).float().cpu().numpy()[0, 0]

            baseline_bands.append(band_power(baseline_out))
            learned_bands.append(band_power(learned_out))
            gains.append(learned_out.std() / baseline_out.std())

        # Averaged over films first, then the ratio, then the gain divided out
        gain = float(np.mean(gains))
        ratios = (np.mean(learned_bands, axis=0) / np.mean(baseline_bands, axis=0)) / gain ** 2

        rows.append((target, INPUT_SIZE / target, gain, *ratios))
        print("{:<8} {:>7.2f}x {:>8.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(*rows[-1]))

        del baseline, learned
        torch.cuda.empty_cache()

    print("\nLaTeX rows: target & decimation & gain & low & mid & high")

    for row in rows:
        print("{} & {:.2f}$\\times$ & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(*row))


if __name__ == "__main__":
    main()
