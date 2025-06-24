import argparse
import random
from pathlib import Path

def main():
    p = argparse.ArgumentParser(
        description="Make an 80/20 train–test split CSV for all .jpgs in a folder"
    )
    p.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Path to folder containing .jpg files (default: current directory)",
    )
    args = p.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
        exit(1)

    # gather all .jpg files (non-recursive)
    jpg_paths = [p for p in folder.glob("*.jpg") if p.is_file()]
    total = len(jpg_paths)
    if total == 0:
        print(f"No .jpg files found in {folder}.")
        return

    # compute split point
    train_count = total * 8 // 10

    # shuffle in place
    random.shuffle(jpg_paths)

    # write CSV
    out_csv = Path.cwd() / "pcb-split.csv"
    with out_csv.open("w", newline="") as f:
        f.write("object,split,label,image,mask,category\n")
        for idx, img_path in enumerate(jpg_paths, start=1):
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