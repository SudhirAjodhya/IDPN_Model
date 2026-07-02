import os, sys
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# JSRT dataset constants
JSRT_IMAGE_SIZE_PIXELS = 2048
JSRT_PIXEL_SIZE_MM = 0.175
JSRT_PIXELS_PER_MM = 1.0 / JSRT_PIXEL_SIZE_MM

# JSRT Directory config
IMAGE_DIRECTORY = "data_sets/JSRT/images/images"
METADATA_CSV  = "data_sets/JSRT/jsrt_metadata.csv"
INPUT_SIZE = 512 # configure this as necessary -> this is the input size parameter that the image will be fed to the model


@dataclass
class Annotation:
    """
    Annotation fields for annotation of a nodule in original image space
    """

    center_x: float
    center_y: float
    size_mm: float
    subtlety: int # 1 (very subtle), 5 (obvious) 
    state: str


class JSRTDataset(Dataset):
    """
    JSRT Pulmonary Nodule Dataset - 247 posteroanterior CXRs (154 w nodule, rest without)
    Annotations: nodule center (x, y) in original pixel space + diamter in mm

    Args:
        image_dir: Path to images
        metadata_csv: Path to meta data
        input_size: square size to resize image to (512 pixels for e.g)
        transforms: Optional Transforms applied to PIL image before tensor conversion (Wont include ToTenor)
        ?POTENTIALLY: Add parameter for Gaus noise - do some data augmentation

        split_indicies: Optional lit of integer indicies for k-fold subsets. 

    """

    def __init__(self, image_directory: str | Path, metadata_csv: str | Path, input_size: int, transforms: Optional[Callable] = None, split_indices: Optional[List[int]] = None):
        self.image_directory = Path(image_directory)
        self.input_size = input_size
        self.transforms = transforms

        metadata = pd.read_csv(metadata_csv)
        self._nodule_map: Dict[str, List[Annotation]] = self._build_nodule_map(metadata)

        all_images = sorted(self.image_directory.glob("*.png"))
        if len(all_images) == 0: raise FileNotFoundError(f"No PNG images found in {self.image_directory}")

        self._all_samples: List[Path] = all_images
        self._samples = (
            [self._all_samples[i] for i in split_indices]
            if split_indices is not None
            else self._all_samples
        )
 
        # Scale factor: 2048 -> input_size
        self._scale = input_size / JSRT_IMAGE_SIZE_PIXELS


    # Helpers
    def _build_nodule_map(self, metadata: pd.DataFrame) -> Dict[str, List[Annotation]]:
        nodule_map: Dict[str, List[Annotation]] = {}

        nodule_rows = metadata[metadata["subtlety"].notna()] # only process those rows in the csv that have noduels

        for _, row in nodule_rows.iterrows():
            image_name = row["study_id"]
            
            annotation = Annotation(center_x=float(row["x"]), center_y=float(row["y"]), size_mm=float(row["size"]), subtlety=int(row["subtlety"]), state=str(row["state"]))

            nodule_map.setdefault(image_name, []).append(annotation)
        
        return nodule_map


    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load JSRT image as a grayscale uint8 array

        !WE SHOULD VERIFY THIS!!!!!!!!
        These PNGs are stored in palette mode (P) with a descending ramp
        (palette index 0 -> RGB(255,255,255), index 255 -> RGB(0,0,0)).
        Calling .convert("L") applies this palette and produces the correct   
        radiological convention: air=dark, tissue=bright.
 
        Confirmed via:
        - Visual inspection against original JSRT radiographs
        - Palette inspection: img.getpalette()[:3] == [255, 255, 255]
        - Pixel range: uint8, min=0, max=255
        """

        return np.array(Image.open(image_path).convert("L"))
    

    def _annotation_to_bounding_box(self, annotation: Annotation) -> Tuple[float, float, float, float]:
        """
        Convert centre-point + diametr to bounding box, scaled to input size

        Original coords (2048), scaled by self.scale
        Return (x1, y1, x2, y2), clamped to [0, input_size]
        """

        radius = (annotation.size_mm * JSRT_PIXELS_PER_MM) / 2.0  #! Double check this - due to pixel to mm conversion

        x1 = (annotation.center_x - radius) * self._scale
        y1 = (annotation.center_y - radius) * self._scale
        x2 = (annotation.center_x + radius) * self._scale
        y2 = (annotation.center_y + radius) * self._scale

        x1 = max(0.0, x1);  y1 = max(0.0, y1)
        x2 = min(float(self.input_size), x2)
        y2 = min(float(self.input_size), y2)

        return x1, y1, x2, y2
    

    #------------------------------------------------------------
    #   Dataset interface
    #------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        image_path = self._samples[index]
        cxr_name = image_path.name

        # laod as greyscale, resize to input size
        cxr_array = self._load_image(image_path)
        image = Image.fromarray(cxr_array, mode="L")
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        if self.transforms is not None: image = self.transforms(image)

        # (1, H, W) float 32 in [0, 1]
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0

        annotations = self._nodule_map.get(cxr_name, [])

        if len(annotations) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            subtleties = torch.zeros((0,), dtype=torch.long)
            sizes_mm = torch.zeros((0,), dtype=torch.float32)
        else:
            box_list, label_list, subtlety_list, size_list = [], [], [], []

            for annotation in annotations:
                x1, y1, x2, y2 = self._annotation_to_bounding_box(annotation)
                box_list.append([x1, y1, x2, y2])
                label_list.append(1)
                subtlety_list.append(annotation.subtlety)
                size_list.append(annotation.size_mm)
 
            boxes = torch.tensor(box_list, dtype=torch.float32)
            labels = torch.tensor(label_list, dtype=torch.long)
            subtleties = torch.tensor(subtlety_list, dtype=torch.long)
            sizes_mm = torch.tensor(size_list, dtype=torch.float32)        

        
        target = {
            "boxes": boxes,
            "labels": labels,
            "subtleties": subtleties,
            "sizes_mm": sizes_mm,
            "image_id": cxr_name,
            "has_nodule": torch.tensor(len(annotations) > 0, dtype=torch.bool)
        }

        return image_tensor, target


    def get_all_indicies(self) -> List[int]:
        return list(range(len(self._all_samples)))
    

    def summary(self) -> None:
        nodule_count = sum(1 for p in self._samples if p.name in self._nodule_map)
        negative_count = len(self._samples) - nodule_count

        print(
            f"JSRTDataset | total={len(self._samples)} "
            f"| nodule={nodule_count} | negative={negative_count} "
            f"| input_size={self.input_size}px"
        )


def loading_visualization_comparison(image_path: str | Path, save_path: Optional[str | Path] = None) -> None:
    """
    Show CXR Images and conversion methods like .convert("L") -> JSRT images in kaggle datast are "P" (pallete/index color), whereas "L" is usually standard greyscale
    """

    image_path = Path(image_path)
    
    old_image = np.array(Image.open(image_path).convert("L")) # direct palette conversion (potentially wrong
    
    # new approach - numpy roundtrip (raw pallete indicies = actaull pixel values)
    raw = Image.open(image_path)
    holder_array = np.array(raw)
    new_image = np.array(Image.fromarray(holder_array, mode="L"))

    # Difference map visualizaiton
    difference =  old_image.astype(np.int16) - new_image.astype(np.int16)

    plot, axes = plt.subplots(2, 3, figsize=(18,10))
    plot.suptitle(f"Image Loading Comparison - { image_path.name }\n"f"Old: Image.convert('L') | New: numpy roundtrip via fromarray", fontsize=12)

    # Row 0: images
    axes[0, 0].imshow(old_image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"OLD: .convert('L')\nmin={old_image.min()}  max={old_image.max()}  "f"mean={old_image.mean():.1f}", fontsize=10)
    axes[0, 0].axis("off")
 
    axes[0, 1].imshow(new_image, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"NEW: fromarray(numpy roundtrip)\nmin={new_image.min()}  "f"max={new_image.max()}  mean={new_image.mean():.1f}", fontsize=10)
    axes[0, 1].axis("off")
 
    diff_abs = np.abs(difference)
    im = axes[0, 2].imshow(diff_abs, cmap="hot", vmin=0, vmax=max(1, diff_abs.max()))
    axes[0, 2].set_title(f"Absolute difference Heatmap\nmax_diff={diff_abs.max()}  "f"identical={diff_abs.max() == 0}", fontsize=10)
    axes[0, 2].axis("off")
    plot.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
 
    # Row 1: histograms
    axes[1, 0].hist(old_image.ravel(), bins=256, range=(0, 255), color="steelblue", alpha=0.8)
    axes[1, 0].set_title("OLD pixel histogram", fontsize=10)
    axes[1, 0].set_xlabel("Pixel value"); axes[1, 0].set_ylabel("Count")
 
    axes[1, 1].hist(new_image.ravel(), bins=256, range=(0, 255), color="darkorange", alpha=0.8)
    axes[1, 1].set_title("NEW pixel histogram", fontsize=10)
    axes[1, 1].set_xlabel("Pixel value"); axes[1, 1].set_ylabel("Count")
 
    axes[1, 2].hist(difference.ravel(), bins=64, color="crimson", alpha=0.8)
    axes[1, 2].set_title("Difference histogram (old - new)", fontsize=10)
    axes[1, 2].set_xlabel("Pixel difference"); axes[1, 2].set_ylabel("Count")
    axes[1, 2].axvline(0, color="black", linestyle="--", linewidth=1)
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
 
    plt.show()
    

def visualize_sample_with_box(dataset: JSRTDataset, index: int, save_path: Optional[str | Path] = None) -> None:
    """
    Dispalys a dataset sample resizd to input_size with bounding box overlaid

    Confirms:
    - mm -> pixel conversion is correct
    - bbox is centred on the nodule
    - box size looks reasonable relative to the image
 
    Args:
        dataset: JSRTDataset instance.
        index: Index of the sample to visualize.
        save_path: If provided, saves the figure to this path

    """

    image_tensor, target = dataset[index]
    image_numpy = (image_tensor.squeeze().numpy() * 255).astype(np.uint8)

    plot, axes = plt.subplots(1, 1, figsize=(7, 7))
    axes.imshow(image_numpy, cmap="gray", vmin=0, vmax=255)
    axes.set_title(f"{target['image_id']} | inputsize = { dataset.input_size }px"f"| had_nodule={target['has_nodule'].item()}", fontsize=10)

    for index, box in enumerate(target["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1
        rect = mpatches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        axes.add_patch(rect)
        axes.text(x1, y1 - 4, f"{target['sizes_mm'][index].item():.1f}mm | sub={target['subtleties'][index].item()}", color="red", fontsize=8, bbox=dict(facecolor="black", alpha=0.5, pad=1))
 
    axes.axis("off")
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
 
    plt.show()




def resolve_image_number(input_image_number) -> str:
    if input_image_number.isdigit() and int(input_image_number) > 0:
        if int(input_image_number) < 10: return "00" + str(input_image_number)
        if int(input_image_number) < 100: return "0" + str(input_image_number)
        if int(input_image_number) < 1000: return str(input_image_number)

    return "ERROR"
    

if __name__ == "__main__":
    x_ray_type = input("Please type 'T' if you would like to view an X-Ray with a nodule: ")
    input_image_number = input("Please select the image number that you would like to view: ")

    resolved_image_number = resolve_image_number(input_image_number)

    # ensure a valid number has been entered    
    if resolved_image_number == "ERROR":
        print("INVALID: input image number is not an image number or is invalid")
        sys.exit(1)

    # Select X-ray with nodule or without nodule
    if x_ray_type == "T": 
        image = Path(IMAGE_DIRECTORY) / f"JPCLN{resolved_image_number}.png"
    else:
        image = Path(IMAGE_DIRECTORY) / f"JPCNN{resolved_image_number}.png"

    if not image.exists(): 
        print("INVALID: THIS IMAGE DOES NOT EXIST")
        sys.exit(1)

    print(f"Running loading comaprison on : {image.name}")

    # Test
    #loading_visualization_comparison(image)
    #img = Image.open("data_sets/JSRT/images/images/JPCLN001.png")
    #print(img.mode)
    #print(img.palette.mode)  # what colour space is the palette in?
    #palette = img.getpalette()  # returns flat list of 256*3 RGB values
    #print(palette[:30])  # first 10 RGB entries


    loading_visualization_comparison(image)
    dataset = JSRTDataset(IMAGE_DIRECTORY, METADATA_CSV, input_size=INPUT_SIZE)
    dataset.summary()

    img, target = dataset[0]
    print(f"\nSample 0:")
    print(f"image shape : {img.shape}")
    print(f"image range : [{img.min():.3f}, {img.max():.3f}]")
    print(f"boxes       : {target['boxes']}")
    print(f"sizes_mm    : {target['sizes_mm']}")
    print(f"subtleties  : {target['subtleties']}")
 


    visualize_sample_with_box(dataset, index=int(input_image_number) - 1, save_path="outputs/sample_bbox.png")
 
    # for i in range(len(dataset)):
    #     _, t = dataset[i]

    #     if not t["has_nodule"]:
    #         print(f"\nNegative sample at index {i}: {t['image_id']}")
    #         visualize_sample_with_box(dataset, index=i, save_path="outputs/sample_negative.png")
    #         break




