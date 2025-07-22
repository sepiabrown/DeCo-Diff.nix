import argparse
import random
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser(
        description="Make an 80/20 train–test split CSV for all extensions in a folder"
    )
    p.add_argument(
        "--input-dir",
        default=".",
        help="Path to folder containing .jpg files (default: current directory)",
    )
    p.add_argument(
        "--output-dir",
        default=".",
        help="Path to folder containing .jpg files (default: current directory)",
    )
    p.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Split ratio for train and test (default: 0.8)",
    )
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.", file=sys.stderr)
        exit(1)

    # gather all .jpg and .png files (non-recursive)
    jpg_png_paths = [p for ext in ("*.jpg", "*.png") for p in input_dir.glob(ext) if p.is_file()]
    total = len(jpg_png_paths)
    if total == 0:
        print(f"No .jpg or .png files found in {input_dir}.")
        return

    # compute split point
    train_count = int(total * args.split_ratio)

    # shuffle in place
    random.shuffle(jpg_png_paths)

    # write CSV
    out_csv = Path(args.output_dir) / "pcb-split.csv"
    with out_csv.open("w", newline="") as f:
        f.write("object,split,label,image,mask,category\n")
        for idx, img_path in enumerate(jpg_png_paths, start=1):
            split = "train" if idx <= train_count else "test"
            # make image path relative to cwd, or absolute if you prefer
            img_str = str(img_path)
            f.write(f"pcb,{split},normal,{img_str},,good\n")

    print(
        f"Wrote {total} entries to {out_csv.name}: "
        f"{train_count} train, {total - train_count} test."
    )

if __name__ == "__main__":
    main()