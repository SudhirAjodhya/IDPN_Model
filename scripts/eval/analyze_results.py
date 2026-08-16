import argparse
import json
from pathlib import Path
import sys
from pathlib import Path

# Add root directory to path to allow importing from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.run_summary import generate_run_summary

def parse_args():
    p = argparse.ArgumentParser(description="Generate run summary from output directory")
    p.add_argument("--run-group-dir", type=str, required=True, help="Path to the output directory of a run group (e.g. outputs/v8/2026-07-19_smoke)")
    return p.parse_args()

def main():
    args = parse_args()
    
    dir_path = Path(args.run_group_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory {dir_path} does not exist.")
        return

    generate_run_summary(dir_path)

if __name__ == "__main__":
    main()
