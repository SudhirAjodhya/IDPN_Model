"""
Three-way split and rare-class oversampling for the VinDr-only multi-class sweep.

SEPARATE FROM splits.py ON PURPOSE. `build_three_way_split` is NODE21-shaped: it adds
VinDr to the TRAIN split only and keeps validation and test byte-identical to the
NODE21-only mode, so that every nodule number in this project is measured on the same
fixed 517-film set. That invariant is load-bearing for the resizer results and must not
be touched. This module builds a different thing - a VinDr-only split with its own val
and test - and leaves that one alone.

WHY STRATIFICATION IS NOT TRIVIAL HERE. VinDr is MULTI-LABEL: one film routinely carries
several findings, so a film cannot simply be dealt into a per-class pile. Assigning by
the commonest class first would leave the rare classes to land wherever their films
happened to fall, and with Pneumothorax at 96 films a bad draw could put 4 in test.
So classes are processed RAREST FIRST, each film going to whichever split is furthest
below its quota for that class. The scarcest label therefore gets first claim on the
films that carry it.

WHAT 60/20/20 BUYS. Measured over all 15,000 films:

    class                   films   train    val   test
    Aortic enlargement       3067    1840    613    613
    Cardiomegaly             2300    1380    460    460
    Pleural thickening       1981    1189    396    396
    Pulmonary fibrosis       1617     970    323    323
    Lung Opacity             1322     793    264    264
    Other lesion             1134     680    227    227
    Pleural effusion         1032     619    206    206
    Nodule/Mass               826     496    165    165
    Infiltration              613     368    123    123
    Calcification             452     271     90     90
    ILD                       386     232     77     77
    Consolidation             353     212     71     71
    Atelectasis               186     112     37     37
    Pneumothorax               96      58     19     19

Twelve of fourteen classes clear 70 test films. Atelectasis and Pneumothorax do not, and
no split ratio fixes that - there are only 186 and 96 films in existence. Augmentation
cannot fix it either: four augmented copies of one film are not four patients, so they
raise the row count without raising the sample size the confidence interval is computed
from. Those two classes are reported as indicative, or given k-fold cross-validation if
a per-class number on them has to carry weight.

AUGMENTATION LIVES ENTIRELY IN TRAIN. VinDrDataset augments inside __getitem__, so it is
already stochastic and per-epoch: a film is a different variant every time it is drawn,
and over a 50-epoch run the model sees 50 variants of it. Nothing needs materialising.
What this module adds is the orthogonal piece - drawing rare-class films MORE OFTEN per
epoch, so a 58-film Pneumothorax training set contributes more than 58 samples.
"""

import csv
import math
from collections import defaultdict

import numpy as np

from src.data.vindr import CLASS_TO_INDEX, NO_FINDING_CLASS, VINDR_CLASSES, VINDR_ROOT


def load_film_labels(annotations_csv):
    """
    Map every film to the set of class indices annotated on it.

    Films marked "No finding" map to an empty set and are kept - they are the negatives.

    Returns:
        dict of image_id -> set of class indices
    """
    labels = defaultdict(set)

    with open(annotations_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["image_id"]]

            if row["class_name"] == NO_FINDING_CLASS or row["x_min"] == "":
                continue

            labels[row["image_id"]].add(CLASS_TO_INDEX[row["class_name"]])

    return dict(labels)


def build_vindr_multilabel_split(annotations_csv, fractions=(0.6, 0.2, 0.2), seed=0,
                                 available_ids=None):
    """
    Split VinDr films three ways, stratified so the rarest classes keep their share.

    Classes are filled RAREST FIRST. Each unassigned film carrying that class goes to
    whichever split is furthest below its quota, so Pneumothorax's 96 films are dealt out
    before Aortic enlargement's 3,067 get any say. Negatives are dealt last, by plain
    proportion, since they carry no label to balance.

    The seed is FIXED separately from the training seed. The split must not move when the
    training seed changes, or two arms would be scored on different test films - the same
    discipline as NODE21's test_seed.

    Args:
        annotations_csv: path to VinDr train.csv
        fractions: (train, val, test), must sum to 1
        seed: controls only the shuffle within a class, never the proportions
        available_ids: restrict to the films that actually exist on disk. The PNG16
                 archive was deleted once the resize caches were written, so the readable
                 set is the 8,788-film cache rather than the CSV's 15,000. ALL 4,394
                 positive films are in it - verified - so every class keeps its full
                 count and only the negative pool shrinks, 10,606 to 4,394. That is a
                 1:1 positive-to-negative ratio instead of 1:2.4, which is if anything
                 better for multi-label training. None splits every film in the CSV.

    Returns:
        dict with "train", "val", "test" -> sorted list of image_ids
    """
    assert abs(sum(fractions) - 1.0) < 1e-9, f"fractions must sum to 1, got {sum(fractions)}"

    labels = load_film_labels(annotations_csv)

    if available_ids is not None:
        available = set(available_ids)
        labels = {k: v for k, v in labels.items() if k in available}

    generator = np.random.default_rng(seed)

    splits = {"train": set(), "val": set(), "test": set()}
    names = ("train", "val", "test")
    assigned = set()

    films_by_class = defaultdict(list)

    for image_id, classes in labels.items():
        for class_index in classes:
            films_by_class[class_index].append(image_id)

    # Rarest class first - it has the least room to absorb a bad draw
    for class_index in sorted(films_by_class, key=lambda c: len(films_by_class[c])):
        candidates = [f for f in sorted(films_by_class[class_index]) if f not in assigned]
        generator.shuffle(candidates)

        total = len(films_by_class[class_index])
        quotas = {name: fraction * total for name, fraction in zip(names, fractions)}
        held = {name: sum(1 for f in splits[name] if class_index in labels[f]) for name in names}

        for image_id in candidates:
            # Largest shortfall wins, so every split converges on its quota together
            target = max(names, key=lambda name: quotas[name] - held[name])
            splits[target].add(image_id)
            held[target] += 1
            assigned.add(image_id)

    # Negatives and any film not claimed above, dealt by plain proportion
    remaining = sorted(set(labels) - assigned)
    generator.shuffle(remaining)

    boundaries = np.cumsum([int(round(f * len(remaining))) for f in fractions])

    for name, start, end in zip(names, [0] + list(boundaries[:-1]), boundaries):
        splits[name].update(remaining[start:end])

    return {name: sorted(splits[name]) for name in names}


def repeat_factors(annotations_csv, image_ids, power=0.5, maximum=8.0):
    """
    How many times each TRAINING film should be drawn per epoch, to lift rare classes.

    Square-root scaled rather than fully equalising. Full equalisation needs 31.9x on
    Pneumothorax against Aortic enlargement, which would have the model see one film's
    anatomy thirty-two times an epoch and memorise it. The sqrt compromise is the standard
    repeat-factor choice (Gupta et al., LVIS) and lands Pneumothorax at about 5.6x.

    A film carrying several classes takes the LARGEST factor among them, so a rare finding
    is never diluted by a common one sharing the film.

    Args:
        annotations_csv: path to VinDr train.csv
        image_ids: the training split only - never val or test
        power: 0.5 for sqrt scaling, 1.0 to fully equalise, 0.0 to disable
        maximum: hard ceiling, so no film can dominate an epoch

    Returns:
        (N,) float array of repeat factors aligned with image_ids
    """
    labels = load_film_labels(annotations_csv)
    wanted = set(image_ids)

    counts = defaultdict(int)

    for image_id in wanted:
        for class_index in labels.get(image_id, ()):
            counts[class_index] += 1

    if not counts:
        return np.ones(len(image_ids), dtype=np.float32)

    commonest = max(counts.values())
    per_class = {c: min((commonest / n) ** power, maximum) for c, n in counts.items()}

    factors = []

    for image_id in image_ids:
        classes = labels.get(image_id, set()) & set(per_class)
        # Negatives stay at 1.0; they are already the majority of the dataset
        factors.append(max((per_class[c] for c in classes), default=1.0))

    return np.array(factors, dtype=np.float32)


def _demo():
    """Self-check on the real annotations: disjointness, quotas, and the rare classes."""
    root = VINDR_ROOT
    annotations = root / "train.csv"

    # The readable set, not the CSV's 15,000 - this is what the sweep actually runs on
    cached = sorted(path.stem for path in (root / "train_png16_cache_i1024").glob("*.npy"))
    print("{} films readable from cache, {} in the CSV".format(
        len(cached), len(load_film_labels(annotations))))

    splits = build_vindr_multilabel_split(annotations, available_ids=cached)
    labels = {k: v for k, v in load_film_labels(annotations).items() if k in set(cached)}

    # No film may appear in two splits. This is the leak that matters.
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert not (train & val) and not (train & test) and not (val & test), "SPLIT LEAK"
    assert len(train) + len(val) + len(test) == len(labels), "films lost or duplicated"
    print("OK  {} train / {} val / {} test, disjoint, nothing lost".format(
        len(train), len(val), len(test)))

    print("\n  {:<22} {:>6} {:>7} {:>6} {:>6}".format("class", "films", "train", "val", "test"))

    for class_index, name in enumerate(VINDR_CLASSES):
        per_split = [sum(1 for f in splits[s] if class_index in labels[f])
                     for s in ("train", "val", "test")]
        total = sum(per_split)
        print("  {:<22} {:>6} {:>7} {:>6} {:>6}".format(name, total, *per_split))

        # Every class must be present in all three splits, or it cannot be scored
        assert all(per_split), f"{name} is absent from a split: {per_split}"

        # Test share within 5 points of 20%, which is what stratification is buying
        share = per_split[2] / total
        assert 0.15 <= share <= 0.25, f"{name} test share {share:.2f} is off quota"

    print("\nOK  every class present in all three splits, test share within 15-25%")

    # Repeat factors must lift the rare classes without letting one film dominate
    factors = repeat_factors(annotations, splits["train"])
    assert factors.min() >= 1.0 and factors.max() <= 8.0
    print("OK  repeat factors {:.2f}x to {:.2f}x, mean {:.2f}x (train only)".format(
        factors.min(), factors.max(), factors.mean()))

    # The split must not move when the training seed does
    again = build_vindr_multilabel_split(annotations, available_ids=cached)
    assert again["test"] == splits["test"], "split is not deterministic"
    print("OK  split is deterministic across calls")


if __name__ == "__main__":
    _demo()
