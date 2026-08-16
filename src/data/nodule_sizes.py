"""
Nodule size stratification - Used in the LUNA16 longest-diameter bins

We use RECIST convention (longest axis) in mm. The 10/20mm cutoffs are LUNDA16s; there is no official Node21 size stratifiction.
Therefore we adopt the community default and cite it in the captions
"""

import math

SMALL_NODULE_SIZE_MAX = 10.0
MEDIUM_NODULE_SIZE_MAX = 20.0
BINS = ("S", "M", "L")

def bin_assignment(nodule_diameter_mm: float) -> str:

    if nodule_diameter_mm < SMALL_NODULE_SIZE_MAX: return "S"
    if nodule_diameter_mm < MEDIUM_NODULE_SIZE_MAX: return "M"

    return "L"


def longest_diameter_mm(width_pixels: float, height_pixels: float, pixel_spacing_mm: float) -> float:
    """
    RECIST longest-axis diameter of a bounding bo in mm

    This matches clinical measurement ( longest lesion axis ) and LUNA16 binning. 
    Note: the training loader `NODE21Dataset` already uses max(w,h) in pixels
    for its own reasons; this function does the same computation in mm so
    the stratification cutoffs are physically meaningful
    """

    return max(width_pixels, height_pixels) * pixel_spacing_mm


if __name__ == "__main__":
    # sanity checking
    assert bin_assignment(5.0) == "S"
    assert bin_assignment(9.99) == "S"
    assert bin_assignment(10.0) == "M"
    assert bin_assignment(19.99) == "M"
    assert bin_assignment(20.0) == "L"
    assert bin_assignment(50.0) == "L"
    assert longest_diameter_mm(10, 40, 0.5) == 20.0    # Elongated bbox: 10x40 px @ 0.5 mm/px -> 20 mm longest axis (Large boundary).
    assert longest_diameter_mm(20, 20, 0.2) == 4.0    # Square bbox: 20x20 px @ 0.2 mm/px -> 4 mm.
    print("ok")
