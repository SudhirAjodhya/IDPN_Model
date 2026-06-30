from core.seeding import set_seed
import pytest
import torch
import numpy as np

def test_set_seed_torch_reproducibility():
    """
    Same seed should produce identical random tensors
    """

    set_seed(42)
    t1 = torch.rand(10)

    set_seed(42)
    t2 = torch.rand(10)

    assert torch.allclose(t1, t2), "Seeds not producing identical tensors"


def test_different_seeds_differ():
    """
    Different seeds should (almost certainly) produce different tensors
    """

    set_seed(42)
    t1 = torch.rand(10)

    set_seed(99)
    t2 = torch.rand(10)

    assert not torch.allclose(t1, t2), "Different seeds produced identical tensors"


def test_numpy_reproducibility():
    
    set_seed(42)
    a1 = np.random.rand(10)

    set_seed(42)
    a2 = np.random.rand(10)

    assert np.allclose(a1, a2)