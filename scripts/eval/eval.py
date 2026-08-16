import argparse
import json
import torch
from pathlib import Path

import sys
from pathlib import Path

# Add root directory to path to allow importing from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.detector import NoduleDetector
from src.data.splits import build_three_way_split
from src.evaluation.detection_metrics import compute_detection_metrics_v4
from src.evaluation.froc import compute_froc_data_v5
from src.visualization.froc_curve import save_froc_curve
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data_sets"
JSRT_IMAGE_DIR = DATA_DIR / "JSRT" / "images" / "images"
JSRT_CSV_PATH = DATA_DIR / "JSRT" / "jsrt_metadata.csv"
NODE21_IMAGE_DIR = DATA_DIR / "node21" / "cxr_images" / "original_data" / "images"
NODE21_CSV_PATH = DATA_DIR / "node21" / "cxr_images" / "original_data" / "metadata.csv"

def parse_args():
    p = argparse.ArgumentParser(description="LDM Evaluation Script")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    p.add_argument("--variant", type=str, required=True)
    p.add_argument("--head", choices=["centernet", "centernet_hr"], default="centernet")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-seed", type=int, default=0)
    p.add_argument("--froc", action="store_true")
    p.add_argument("--input-size", type=int, default=2048)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint: {args.checkpoint}")
    model = NoduleDetector(variant=args.variant, head=args.head).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=JSRT_CSV_PATH,
        jsrt_image_dir=JSRT_IMAGE_DIR,
        node21_csv_path=NODE21_CSV_PATH,
        node21_image_dir=NODE21_IMAGE_DIR,
        seed=args.seed,
        test_seed=args.test_seed,
    )
    
    test_dls = {k: DataLoader(v, batch_size=args.batch_size, shuffle=False, num_workers=4) for k, v in test_loaders.items()}
    
    print("Evaluating detection metrics on combined test set...")
    combined_metrics = compute_detection_metrics_v4(model, test_dls["combined"], device=device, input_size=args.input_size)
    
    print(json.dumps(combined_metrics, indent=4))
    
    if args.froc:
        print("Computing FROC data...")
        froc_data = compute_froc_data_v5(model, test_dls["combined"], device=device, input_size=args.input_size)
        out_dir = Path(args.checkpoint).parent
        save_froc_curve(froc_data, out_dir, args.variant)
        with open(out_dir / "froc_data_eval.json", "w") as f:
            json.dump(froc_data, f, indent=4)
        print(f"Saved FROC curve to {out_dir}")

if __name__ == "__main__":
    main()
