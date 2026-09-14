"""
Batch sampler that keeps every batch to a single source dataset.

Why this exists: the model is genuinely input-resolution-agnostic (verified - a
ConvNeXt+FPN detector at target 224 emits (B, 1, 56, 56) from 1024, 2048 and 2700 inputs
alike), but `default_collate` stacks a batch into one (B, 1, H, W) tensor, so every film
in a batch must share H and W. Pooling NODE21 at 1024 with VinDr at its native ~2700
therefore fails at collation, not in the model.

Drawing each batch from one dataset removes that constraint without padding, without
resizing one dataset to match the other, and without touching the model.

The point of running VinDr at native resolution is that it hands the resizer a ~12x
decimation to 224 against NODE21's 4.6x. Measured over 15 films at 2700px native, bilinear
carries 1.32x more aliasing error at 12.1x than at 4.6x under the CLAHE pipeline, so the
harder case gives a learned resizer more to actually repair.

SAFETY NOTE: single-source batches are safe with ConvNeXt because it has 0 BatchNorm and
19 LayerNorm layers, so no running statistics can be poisoned by a domain-homogeneous
batch. This sampler must NOT be used with `resnet50_fpn`, which carries 53 BatchNorm
layers - alternating single-domain batches would put a BN-statistics confound inside the
comparison.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class SingleSourceBatchSampler(Sampler):
    """
    Yield index batches, each drawn entirely from one component of a ConcatDataset.

    Indices are the flat indices a ConcatDataset uses, so component i occupies the
    contiguous range [sum(sizes[:i]), sum(sizes[:i+1])).
    """

    def __init__(self, dataset_sizes: list[int], batch_size: int, shuffle: bool = True,
                 seed: int = 42, drop_last: bool = False):
        """
        Args:
            dataset_sizes: length of each component, in ConcatDataset order
            batch_size: films per batch
            shuffle: shuffle within each component and shuffle the batch order
            seed: base seed; the epoch counter is added so each epoch differs
            drop_last: drop a component's trailing partial batch. Off by default,
                since a partial batch is still single-source and still valid.
        """
        self.dataset_sizes = list(dataset_sizes)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

    def _build_batches(self, epoch: int) -> list[list[int]]:
        """Group each component's indices into batches, then interleave the batches."""
        generator = np.random.default_rng(self.seed + epoch)
        batches = []
        offset = 0

        for size in self.dataset_sizes:
            indices = np.arange(offset, offset + size)

            if self.shuffle:
                generator.shuffle(indices)

            for start in range(0, size, self.batch_size):
                chunk = indices[start:start + self.batch_size].tolist()

                if len(chunk) == self.batch_size or not self.drop_last:
                    batches.append(chunk)

            offset += size

        # Interleave so the optimizer alternates between domains rather than seeing all
        # of one dataset and then all of the other, which would be a curriculum.
        if self.shuffle:
            generator.shuffle(batches)

        return batches

    def __iter__(self):
        batches = self._build_batches(self._epoch)
        self._epoch += 1

        return iter(batches)

    def __len__(self):
        return len(self._build_batches(self._epoch))


def _demo() -> None:
    """Self-check for the property this class exists to guarantee."""

    sizes = [3591, 15000]
    boundary = sizes[0]
    sampler = SingleSourceBatchSampler(sizes, batch_size=8, seed=42)

    batches = list(sampler)

    # THE load-bearing property: no batch may straddle the component boundary, because
    # the two components hold different image sizes and could not be stacked.
    for batch in batches:
        sources = {index >= boundary for index in batch}
        assert len(sources) == 1, f"batch mixes sources: {batch[:5]}"

    # Every sample appears exactly once per epoch, so no film is dropped or duplicated.
    flat = sorted(index for batch in batches for index in batch)
    assert flat == list(range(sum(sizes))), "epoch must cover every index exactly once"

    # len() must match what iteration actually yields, or progress bars and schedulers lie.
    assert len(SingleSourceBatchSampler(sizes, batch_size=8, seed=42)) == len(batches)

    # Batches must interleave rather than run one dataset and then the other. A short
    # leading window is the wrong test: the smaller component is only ~19% of batches, so
    # a 20-batch window legitimately misses it about 1.3% of the time. Test the property
    # directly instead - each source's batches should average near the midpoint, which is
    # false by a wide margin if the order is still concatenated.
    positions = np.array([index for index, batch in enumerate(batches)
                          if batch[0] < boundary])
    midpoint = len(batches) / 2
    assert abs(positions.mean() - midpoint) < 0.15 * len(batches), \
        f"minority source clusters at {positions.mean():.0f}, expected near {midpoint:.0f}"
    assert len({batch[0] >= boundary for batch in batches[:200]}) == 2, \
        "both sources should appear within the first 200 batches"

    # Successive epochs must differ, or shuffling is not happening.
    reshuffled = list(sampler)
    assert reshuffled != batches, "epoch 2 must not repeat epoch 1"

    # drop_last=True must yield only full batches.
    dropped = list(SingleSourceBatchSampler(sizes, batch_size=8, drop_last=True))
    assert all(len(batch) == 8 for batch in dropped)

    print(f"pooled_batch_sampler self-check passed "
          f"({len(batches)} batches over {sum(sizes)} films, none mixed)")


if __name__ == "__main__":
    _demo()
