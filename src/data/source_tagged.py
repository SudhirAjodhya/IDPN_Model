from torch.utils.data import Dataset
from pathlib import Path
import cv2
import torch

class SourceTaggedDataset(Dataset):
    """
    Adds 'source' and (if missing) 'size_mm' to each target dict
    """
    def __init__(self, dataset, source_name):
        self.dataset = dataset
        self.source_name = source_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        target["source"] = self.source_name
        
        if "size_mm" not in target:
            target["size_mm"] = 0.0
        
        return image, target


class LungMaskDataset(Dataset):
    """
    Wraps any dataset and appends a precomputed lung mask to each target dict
    Mask filenames: {source_name}_{img_filename_with_dots_replaced_by_underscores}.png
    Falls back to all-ones mask when the file is missing
    """
    def __init__(self, dataset, source_name, lung_mask_dir, target_size=512):
        self.dataset = dataset
        self.source_name = source_name
        self.mask_dir = Path(lung_mask_dir)
        self.target_size = target_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        fname = target.get("filename", "")
        img_name = Path(fname).name if fname else ""
        mask_path = self.mask_dir / f"{self.source_name}_{img_name.replace('.', '_')}.png"

        if mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (self.target_size, self.target_size),
                           interpolation=cv2.INTER_NEAREST)
            target["lung_mask"] = torch.from_numpy(m).float() / 255.0
        else:
            target["lung_mask"] = torch.ones((self.target_size, self.target_size))

        return image, target
