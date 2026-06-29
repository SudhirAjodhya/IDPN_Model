import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

def print_image_diagnostics(image_sitk: sitk.Image, image_array: np.ndarray) -> None:
    """
    Extracts and logs physical metadata from the MHA header and array properties 
    to verify the image is a standard 2D CXR rather than a CT/MRI slice.
    """
    print("-" * 40)
    print("IMAGE DIAGNOSTICS")
    print("-" * 40)
    print(f"Resolution (W, H): {image_sitk.GetSize()}")
    print(f"Pixel Spacing (mm): {image_sitk.GetSpacing()}")
    print(f"Array Shape: {image_array.shape}")
    print(f"Data Type: {image_array.dtype}")
    print(f"Min Pixel Value: {image_array.min()}")
    print(f"Max Pixel Value: {image_array.max()}")
    print("-" * 40)

def main() -> None:
    """
    Main execution block: Loads a specific CXR, normalizes its bit-depth for 
    OpenCV, maps bounding box coordinates from the dataset metadata, and saves 
    the annotated result for visual verification.
    """

    #  image_path = current_dir / "cxr_images" / "original_data" / "images" / "c0001.mha"
    current_dir =  Path("../../data_sets/node21") #Path(__file__).parent
    target_image_name = "n0239.mha" 
    
    image_path = current_dir / "cxr_images" / "original_data" / "images" / "n0239.mha"
    csv_path = current_dir / "cxr_images" / "original_data" / "metadata.csv"

    if not image_path.exists():
        print(f"Error: Image not found at {image_path.resolve()}")
        sys.exit(1)
        
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path.resolve()}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    image_metadata = df[df['img_name'] == target_image_name]
    
    if image_metadata.empty:
        print(f"Warning: No bounding box annotations found for {target_image_name}.")
    
    image = sitk.ReadImage(str(image_path))
    array = sitk.GetArrayFromImage(image)
    
    print_image_diagnostics(image, array)
    
    # Flatten single-channel medical images to 2D for standard processing
    img_2d = array.squeeze()
    
    # Scale native medical bit-depth to 8-bit unsigned integer for OpenCV rendering
    img_8bit = cv2.normalize(img_2d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Convert to BGR to allow for colored bounding box overlays on the grayscale base
    img_color = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    
    for index, row in image_metadata.iterrows():
        # Map Node21 coordinates (x,y are top-left) to standard bounding box bounds
        x_min = int(row['x'])
        y_min = int(row['y'])
        x_max = int(row['x'] + row['width'])
        y_max = int(row['y'] + row['height'])
        
        cv2.rectangle(img_color, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=3)
        cv2.putText(
            img_color, 
            "Nodule", 
            (x_min, y_min - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 0, 255), 
            thickness=2
        )
        
        print(f"Bounding box mapped: {x_min}, {y_min} -> {x_max}, {y_max}")

    output_path = current_dir / f"visualized_{target_image_name}.png"
    cv2.imwrite(str(output_path), img_color)
    print(f"\nRendered image saved to: {output_path.resolve()}")

if __name__ == "__main__":
    main()