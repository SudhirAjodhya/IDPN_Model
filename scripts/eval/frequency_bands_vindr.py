"""
VinDr version of frequency_bands: does the learned resizer sharpen or blur, per output size?

Same measurement as the NODE21 script, so the two tables are directly comparable. For each
output resolution and each seed, both arms' resizer outputs are computed on the same VinDr
test films (2048 input). Power is averaged over three radial bands, as a fraction of
Nyquist: low [0, 0.1), mid [0.1, 0.4), high [0.4, 1.0). Each learned/bilinear ratio is
divided by the squared contrast gain, so a value of 1 means "a pure contrast stretch of
the bilinear output", above 1 means more of that band (sharpening), below 1 less
(blurring).

Before any model is loaded, band_power is checked on a synthetic image with known filters
(identity, pure scaling, sharpening, blurring). If any check fails the script stops.

Usage (from Model/):
    python -m scripts.eval.frequency_bands_vindr
    python -m scripts.eval.frequency_bands_vindr --films 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval.frequency_bands import band_power                        # noqa: E402
from scripts.eval.report_tables import BASELINE, LEARNED, VINDR_GROUP, VINDR_INPUT  # noqa: E402
from src.data.vindr import VinDrDataset, NUM_VINDR_CLASSES, VINDR_ROOT     # noqa: E402
from src.data.vindr_split import build_vindr_multilabel_split              # noqa: E402
from src.models.detector import NoduleDetector                             # noqa: E402

ANNOTATIONS = VINDR_ROOT / "train.csv"
IMAGES = VINDR_ROOT / "train_png16"


def gain_corrected_ratios(reference_bands, candidate_bands, gain):
    """Band power ratio candidate/reference with the contrast gain divided out."""
    return (np.mean(candidate_bands, axis=0) / np.mean(reference_bands, axis=0)) / gain ** 2


def check_band_power():
    """
    Known filters must give known ratios, or the measurement cannot be trusted.

    The test image is 1/f noise, which has a natural-image-like spectrum.
    """
    generator = np.random.default_rng(0)
    frequency = np.hypot(*np.meshgrid(np.fft.fftfreq(224), np.fft.fftfreq(224)))
    frequency[0, 0] = 1.0
    image = np.real(np.fft.ifft2(np.fft.fft2(generator.standard_normal((224, 224))) / frequency))

    def ratios(candidate):
        gain = candidate.std() / image.std()

        return gain_corrected_ratios([band_power(image)], [band_power(candidate)], gain)

    sharpened = image + 1.5 * (image - ndimage.gaussian_filter(image, 1.0))
    blurred = ndimage.gaussian_filter(image, 1.0)

    assert np.allclose(ratios(image.copy()), 1.0), "identity is not 1"
    assert np.allclose(ratios(3.0 * image), 1.0), "contrast gain not divided out"
    assert ratios(sharpened)[2] > 1.0, "sharpening not detected"
    assert ratios(blurred)[2] < 1.0, "blurring not detected"

    print("band_power checks passed: sharpened high {:.2f}, blurred high {:.2f}".format(
        ratios(sharpened)[2], ratios(blurred)[2]))


def find_runs():
    """(target, variant, seed) -> run directory, for the report's VinDr run group."""
    runs = {}

    for path in (ROOT / "outputs" / "vindr").rglob("metrics.json"):
        metrics = json.load(open(path))

        if metrics.get("run_group") == VINDR_GROUP:
            runs[(metrics["target_size"], metrics["variant"], metrics["seed"])] = (path.parent, metrics)

    return runs


def load_model(run_dir, metrics, device):
    """Rebuild one VinDr run's model from its checkpoint."""
    model = NoduleDetector(variant=metrics["variant"], head="fcos",
                           backbone=metrics.get("backbone", "convnext_fpn"),
                           target_size=metrics["target_size"],
                           num_classes=NUM_VINDR_CLASSES).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--films", type=int, default=100, help="first N films of the test split")
    arguments = parser.parse_args()

    check_band_power()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = find_runs()
    targets = sorted({key[0] for key in runs})
    seeds = sorted({key[2] for key in runs})

    # The fixed VinDr test split, in its own order, unaugmented. Split seed is shared by all runs
    split_seed = next(iter(runs.values()))[1]["split_seed"]
    cache_dir = VINDR_ROOT / "train_png16_cache_i{}".format(VINDR_INPUT)
    available = sorted(path.stem for path in cache_dir.glob("*.npy"))
    splits = build_vindr_multilabel_split(ANNOTATIONS, seed=split_seed, available_ids=available)

    test_set = VinDrDataset(annotations_csv=ANNOTATIONS, images_root=IMAGES,
                            image_ids=splits["test"], augment=False, input_size=VINDR_INPUT,
                            clahe=True, multiclass=True, cache_dir=cache_dir)

    films = []

    for batch in DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0):
        if len(films) >= arguments.films: break

        films.append(batch[0] if isinstance(batch, (list, tuple)) else batch["image"])   # (1, 1, 2048, 2048)

    print("VinDr, input {}, seeds {}, {} test films".format(VINDR_INPUT, seeds, len(films)))
    print("{:<7} {:>7} {:>5} {:>14} {:>14} {:>14} {:>14}".format(
        "target", "decim", "seed", "gain", "low", "mid", "high"))

    table_rows = []

    for target in targets:
        per_seed = []

        for seed in seeds:
            if (target, BASELINE, seed) not in runs or (target, LEARNED, seed) not in runs:
                print("{:<7} seed {} missing run".format(target, seed))
                continue

            baseline = load_model(*runs[(target, BASELINE, seed)], device)
            learned = load_model(*runs[(target, LEARNED, seed)], device)

            baseline_bands, learned_bands, gains = [], [], []

            # Resizer outputs only, in fp32, as in the NODE21 script
            for image in films:
                image = image.to(device)

                with torch.no_grad():
                    baseline_output = baseline.ldm(image).float().cpu().numpy()[0, 0]    # (S, S)
                    learned_output = learned.ldm(image).float().cpu().numpy()[0, 0]

                baseline_bands.append(band_power(baseline_output))
                learned_bands.append(band_power(learned_output))
                gains.append(learned_output.std() / baseline_output.std())

            # Averaged over films first, then the ratio, then the gain divided out
            gain = float(np.mean(gains))
            ratios = gain_corrected_ratios(baseline_bands, learned_bands, gain)
            per_seed.append((gain, *ratios))

            print("{:<7} {:>6.2f}x {:>5} {:>14.2f} {:>14.2f} {:>14.2f} {:>14.2f}".format(
                target, VINDR_INPUT / target, seed, gain, *ratios))

            del baseline, learned
            torch.cuda.empty_cache()

        if not per_seed: continue

        # Mean over seeds, with the observed range, for the table
        values = np.array(per_seed)
        table_rows.append((target, VINDR_INPUT / target, values.mean(axis=0),
                           values.min(axis=0), values.max(axis=0)))

    print("\nLaTeX rows (mean over seeds, range in brackets): target & decimation & gain & low & mid & high")

    for target, decimation, mean, low, high in table_rows:
        cells = ["{:.2f} [{:.2f}, {:.2f}]".format(m, l, h) for m, l, h in zip(mean, low, high)]
        print("{} & {:.0f}$\\times$ & {} \\\\".format(target, decimation, " & ".join(cells)))


if __name__ == "__main__":
    main()
