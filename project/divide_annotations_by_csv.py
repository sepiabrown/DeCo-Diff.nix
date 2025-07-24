import os
import csv
import json
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Divide annotation files based on split CSV references.")
    parser.add_argument('--csv', required=True, help='Path to split CSV file')
    parser.add_argument('--ann_folder', required=True, help='Path to annotation folder')
    parser.add_argument('--out_referenced', required=False, help='Output folder for referenced annotation files')
    parser.add_argument('--out_unreferenced', required=False, help='Output folder for unreferenced annotation files')
    return parser.parse_args()

def get_referenced_images(csv_path):
    referenced = set()
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row.get('image')
            if img_path:
                referenced.add(os.path.normpath(img_path))
    return referenced

def main():
    args = parse_args()

    # Derive default output folder names if not provided
    ann_folder_name = os.path.basename(os.path.normpath(args.ann_folder))
    csv_name = os.path.splitext(os.path.basename(args.csv))[0]
    if args.out_referenced:
        out_referenced = args.out_referenced
    else:
        out_referenced = f"{ann_folder_name}_{csv_name}_referenced"
    if args.out_unreferenced:
        out_unreferenced = args.out_unreferenced
    else:
        out_unreferenced = f"{ann_folder_name}_{csv_name}_unreferenced"

    os.makedirs(out_referenced, exist_ok=True)
    os.makedirs(out_unreferenced, exist_ok=True)

    referenced_images = get_referenced_images(args.csv)

    for fname in os.listdir(args.ann_folder):
        if not fname.endswith('.json'):
            continue
        ann_path = os.path.join(args.ann_folder, fname)
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann = json.load(f)
        image_path = os.path.normpath(ann.get('image_path', ''))
        if image_path in referenced_images:
            shutil.copy2(ann_path, os.path.join(out_referenced, fname))
        else:
            shutil.copy2(ann_path, os.path.join(out_unreferenced, fname))

if __name__ == '__main__':
    main()