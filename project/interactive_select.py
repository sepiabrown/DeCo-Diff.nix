import argparse
import os
import sys
import json
from pathlib import Path
import re
from utils import path_to_safe_filename

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def screen_size():
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h

def ask_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Images", "*.png *.jpg *.tif *.tiff *.bmp")]
    )
    root.destroy()
    return Path(path) if path else None



class GridAnnotationWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.annotations = {}  # Store annotations per image
        out_dir.mkdir(parents=True, exist_ok=True)
        self.load_existing_annotation_files()

    def load_existing_annotation_files(self):
        """Load existing annotation files from the output directory."""
        if not self.out_dir.exists():
            return
            
        for json_file in self.out_dir.glob("*__annotations.json"):
            try:
                with open(json_file, 'r') as f:
                    annotation = json.load(f)
                    image_path = annotation.get("image_path")
                    if image_path:
                        self.annotations[image_path] = annotation
                        print(f"Loaded existing annotations from: {json_file.name}")
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

    def add_defect(self, grid_row: int, grid_col: int, image_path: Path):
        """Add a defective patch at the specified grid position."""
        image_key = str(image_path)
        if image_key not in self.annotations:
            self.annotations[image_key] = {
                "image_path": str(image_path),
                "defective_patches": [],
                "grid_size": 128
            }
        
        # Check if this patch is already marked as defective
        patch_key = (grid_row, grid_col)
        if patch_key not in self.annotations[image_key]["defective_patches"]:
            # Convert to regular Python integers to ensure JSON serialization
            self.annotations[image_key]["defective_patches"].append([int(grid_row), int(grid_col)])
            print(f"     marked defective: grid[{grid_row}, {grid_col}]")

    def remove_defect(self, grid_row: int, grid_col: int, image_path: Path):
        """Remove a defective patch at the specified grid position."""
        image_key = str(image_path)
        if image_key in self.annotations:
            # Convert to regular Python integers for comparison
            patch_to_remove = [int(grid_row), int(grid_col)]
            if patch_to_remove in self.annotations[image_key]["defective_patches"]:
                self.annotations[image_key]["defective_patches"].remove(patch_to_remove)
                print(f"     removed defective: grid[{grid_row}, {grid_col}]")
                
                # If no more defective patches for this image, remove the entire entry
                if not self.annotations[image_key]["defective_patches"]:
                    del self.annotations[image_key]
                    print(f"     removed all annotations for {image_path.name}")

    def save_annotations(self):
        """Save all annotations to JSON file."""
        if not self.annotations:
            print("No annotations to save.")
            return
        
        # Save each image's annotations separately
        for image_key, annotation in self.annotations.items():
            image_path = Path(annotation["image_path"])
            json_filename = f"{path_to_safe_filename(str(image_path))}__annotations.json"
            json_path = self.out_dir / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(annotation, f, indent=2)
            print(f"Saved annotations to: {json_path}")

class CropApp:
    PAN_STEP = 50
    GRID_SIZE = 128

    def __init__(self, image_path: Path, out_dir: Path, patch_size: int):
        self.out_dir = out_dir
        self.patch_size = patch_size
        self.writer = GridAnnotationWriter(out_dir)
        self.defective_patches = set()  # Track defective patches for current image
        self.load_image(image_path)

        self.scr_w, self.scr_h = screen_size()
        self.view_w = min(self.scr_w - 100, self.img_w)
        self.view_h = min(self.scr_h - 120, self.img_h)

        self.off_x = 0
        self.off_y = 0
        self.cur_view = None

        self.win = "Grid Annotator"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.view_w, self.view_h)
        cv2.setMouseCallback(self.win, self.on_mouse)

    def load_image(self, path: Path):
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Cannot read {path}")
        self.path = path
        self.img = img
        self.img_h, self.img_w = img.shape[:2]
        self.defective_patches = set()  # Reset defective patches for new image
        
        # Load existing annotations for this image
        self.load_existing_annotations(path)
        
        print(f"\nLoaded {path.name}: {self.img_w}x{self.img_h}px")
        if self.defective_patches:
            print(f"Loaded {len(self.defective_patches)} existing defective patches")

    def load_existing_annotations(self, image_path: Path):
        """Load existing annotations for the given image."""
        image_key = str(image_path)
        if image_key in self.writer.annotations:
            annotation = self.writer.annotations[image_key]
            for patch in annotation["defective_patches"]:
                grid_row, grid_col = patch
                self.defective_patches.add((grid_row, grid_col))

    def pan(self, dx: int, dy: int):
        self.off_x = np.clip(self.off_x + dx, 0, max(0, self.img_w - self.view_w))
        self.off_y = np.clip(self.off_y + dy, 0, max(0, self.img_h - self.view_h))

    def get_grid_position(self, x: int, y: int) -> tuple[int, int]:
        """Convert pixel coordinates to grid row/column indices."""
        grid_row = y // self.GRID_SIZE
        grid_col = x // self.GRID_SIZE
        return grid_row, grid_col

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.cur_view = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_click(x, y)

    def current_view(self):
        view = self.img[self.off_y:self.off_y + self.view_h,
                        self.off_x:self.off_x + self.view_w].copy()
        
        # Draw grid with 128-pixel spacing
        grid_spacing = self.GRID_SIZE
        grid_color = (100, 100, 100)  # Gray color for grid lines
        grid_thickness = 1
        
        # Calculate grid start positions relative to the view
        start_x = (-self.off_x) % grid_spacing
        start_y = (-self.off_y) % grid_spacing
        
        # Draw vertical grid lines
        for x in range(start_x, self.view_w, grid_spacing):
            cv2.line(view, (x, 0), (x, self.view_h), grid_color, grid_thickness)
        
        # Draw horizontal grid lines
        for y in range(start_y, self.view_h, grid_spacing):
            cv2.line(view, (0, y), (self.view_w, y), grid_color, grid_thickness)
        
        # Highlight defective patches
        for grid_row, grid_col in self.defective_patches:
            # Convert grid position to pixel coordinates
            pixel_x = grid_col * self.GRID_SIZE
            pixel_y = grid_row * self.GRID_SIZE
            
            # Convert to view coordinates
            view_x = pixel_x - self.off_x
            view_y = pixel_y - self.off_y
            
            # Only draw if the patch is visible in the current view
            if (0 <= view_x < self.view_w and 0 <= view_y < self.view_h and
                view_x + self.GRID_SIZE > 0 and view_y + self.GRID_SIZE > 0):
                
                # Clip to view boundaries
                x1 = max(0, view_x)
                y1 = max(0, view_y)
                x2 = min(self.view_w, view_x + self.GRID_SIZE)
                y2 = min(self.view_h, view_y + self.GRID_SIZE)
                
                # Draw red rectangle for defective patches
                cv2.rectangle(view, (x1, y1), (x2-1, y2-1), (0, 0, 255), 2)
        
        # Show preview rectangle for current mouse position
        if self.cur_view is not None:
            vx, vy = self.cur_view
            X, Y = self.off_x + vx, self.off_y + vy
            grid_row, grid_col = self.get_grid_position(X, Y)
            
            # Calculate patch boundaries
            patch_x = grid_col * self.GRID_SIZE
            patch_y = grid_row * self.GRID_SIZE
            
            # Convert to view coordinates
            view_patch_x = patch_x - self.off_x
            view_patch_y = patch_y - self.off_y
            
            # Only draw if the patch is visible
            if (0 <= view_patch_x < self.view_w and 0 <= view_patch_y < self.view_h and
                view_patch_x + self.GRID_SIZE > 0 and view_patch_y + self.GRID_SIZE > 0):
                
                # Clip to view boundaries
                x1 = max(0, view_patch_x)
                y1 = max(0, view_patch_y)
                x2 = min(self.view_w, view_patch_x + self.GRID_SIZE)
                y2 = min(self.view_h, view_patch_y + self.GRID_SIZE)
                
                # Draw green rectangle for preview
                color = (0, 255, 0) if (grid_row, grid_col) not in self.defective_patches else (0, 255, 255)
                cv2.rectangle(view, (x1, y1), (x2-1, y2-1), color, 2)
        
        return view

    def handle_click(self, x, y):
        X = self.off_x + x
        Y = self.off_y + y
        
        # Get grid position
        grid_row, grid_col = self.get_grid_position(X, Y)
        
        # Toggle defective status
        patch_key = (grid_row, grid_col)
        if patch_key in self.defective_patches:
            self.defective_patches.remove(patch_key)
            self.writer.remove_defect(grid_row, grid_col, self.path)
            print(f"\nUnmarked grid[{grid_row}, {grid_col}] as normal")
        else:
            self.defective_patches.add(patch_key)
            self.writer.add_defect(grid_row, grid_col, self.path)
            print(f"\nMarked grid[{grid_row}, {grid_col}] as defective")

    def run(self):
        print("\nControls:")
        print("  left-click - toggle patch as defective/normal")
        print("  arrow / W/A/X/D - pan")
        print("  o            - open another file")
        print("  s            - save annotations")
        print("  q / ESC      - quit\n")

        while True:
            _, _, win_w, win_h = cv2.getWindowImageRect(self.win)
            self.view_w = min(win_w, self.img_w)
            self.view_h = min(win_h, self.img_h)
            cv2.imshow(self.win, self.current_view())
            cv2.setWindowTitle(
                self.win,
                f"{self.path.name}  [{self.off_x}:{self.off_x+self.view_w},"
                f" {self.off_y}:{self.off_y+self.view_h}] - Defects: {len(self.defective_patches)}")

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                self.writer.save_annotations()
            elif key in (ord('w'), 82):
                self.pan(0, -self.PAN_STEP)
            elif key in (ord('x'), 84):
                self.pan(0, self.PAN_STEP)
            elif key in (ord('a'), 81):
                self.pan(-self.PAN_STEP, 0)
            elif key in (ord('d'), 83):
                self.pan(self.PAN_STEP, 0)
            elif key in (ord('o'), ord('O')):
                new_path = ask_file()
                if new_path:
                    self.load_image(new_path)
                    self.off_x = self.off_y = 0

        # Save annotations before closing
        self.writer.save_annotations()
        cv2.destroyAllWindows()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--image",
                   help="Image to start with (leave empty for open-file dialog)")
    p.add_argument("-o", "--output_dir", required=True,
                   help="Directory to save annotations")
    p.add_argument("-s", "--patch_size", type=int, default=128,
                   help="Square patch side (pixels, default: 128)")
    args = p.parse_args()

    if args.image:
        first_img = Path(args.image)
        if not first_img.exists():
            sys.exit(f"  {first_img} not found")
    else:
        first_img = ask_file()
        if first_img is None:
            sys.exit("No image chosen, bye!")

    app = CropApp(first_img, Path(args.output_dir), args.patch_size)
    app.run()

if __name__ == "__main__":
    main()
