"""
view_dicom.py
=============
Utility script to load and visualize a 2D DICOM image with Matplotlib.

Usage:
    python scripts/diagnostics/view_dicom.py <path_to_dicom_file>
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt


def view_dicom(file_path):
    try:
        import pydicom
    except ImportError:
        print("ERROR: pydicom is not installed. Run: pip install pydicom")
        return

    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found at {path}")
        return

    try:
        dicom_data = pydicom.dcmread(str(path))
        image_array = dicom_data.pixel_array

        plt.figure(figsize=(8, 8))
        plt.imshow(image_array, cmap="gray")
        modality = dicom_data.get("Modality", "Unknown")
        plt.title(f"Modality: {modality} | Shape: {image_array.shape} | File: {path.name}")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"Error loading DICOM file: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        view_dicom(sys.argv[1])
    else:
        print("Usage: python scripts/diagnostics/view_dicom.py <path_to_dicom_file>")
