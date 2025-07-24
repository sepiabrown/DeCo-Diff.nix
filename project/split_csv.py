import argparse
import random
from pathlib import Path
import sys
import re

def natural_key(string):
    # Split string into list of strings and integers for natural sorting
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string)]

def main():
    p = argparse.ArgumentParser(
        description="Make an 80/20 train–test split CSV for all extensions in a folder"
    )
    p.add_argument(
        "--input-dir",
        default=".",
        help="Path to folder containing the files to split (default: current directory)",
    )
    p.add_argument(
        "--output-dir",
        help="Path to folder that contains the output csv file (default: current directory)",
    )
    p.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Split ratio for train and test (default: 0.8)",
    )
    p.add_argument(
        "--extensions",
        nargs="+",
        default=["*"],
        help="File extensions to include (e.g. jpg png bmp). Default: all files",
    )
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.", file=sys.stderr)
        exit(1)

    # Set default output_dir to input_dir if not provided
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    # gather all files with the given extensions (non-recursive)
    if args.extensions == ["*"]:
        file_paths = [p for p in input_dir.iterdir() if p.is_file()]
    else:
        file_paths = []
        for ext in args.extensions:
            ext = ext if ext.startswith(".") else f".{ext}"
            file_paths.extend(input_dir.glob(f"*{ext}"))
        file_paths = [p for p in file_paths if p.is_file()]
    total = len(file_paths)
    if total == 0:
        print(f"No files with extensions {args.extensions} found in {input_dir}.")
        return

    # compute split point
    train_count = int(total * args.split_ratio)

    # shuffle in place
    #random.shuffle(file_paths)
    file_paths.sort(key=lambda p: natural_key(str(p)))
    
    # write CSV
    out_csv = output_dir / "pcb-split.csv"
    with out_csv.open("w", newline="") as f:
        f.write("object,split,label,image,mask,category\n")
        for idx, img_path in enumerate(file_paths, start=1):
            split = "train" if idx <= train_count else "test"
            img_str = str(img_path)
            f.write(f"pcb,{split},normal,{img_str},,good\n")

    print(
        f"Wrote {total} entries to {out_csv.name}: "
        f"{train_count} train, {total - train_count} test."
    )

if __name__ == "__main__":
    main()