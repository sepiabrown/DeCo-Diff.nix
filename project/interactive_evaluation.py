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
        title="Select an evaluation records JSON",
        filetypes=[("JSON files", "*.json")]
    )
    root.destroy()
    return Path(path) if path else None



class EvaluationRecordsWriter:
    def __init__(self, evaluation_records_path: Path, out_dir: Path):
        self.evaluation_records_path = evaluation_records_path
        self.out_dir = out_dir
        self.original_records = {}
        self.selected_records = {}  # Records that are still selected (not eliminated)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.load_evaluation_records()

    def load_evaluation_records(self):
        """Load evaluation records from JSON file."""
        try:
            with open(self.evaluation_records_path, 'r') as f:
                data = json.load(f)
                if "records" in data:
                    for record in data["records"]:
                        record_id = record["record_id"]
                        self.original_records[record_id] = record
                        self.selected_records[record_id] = record  # Initially all are selected
                    print(f"Loaded {len(self.original_records)} evaluation records")
                else:
                    print("Warning: No 'records' field found in JSON")
        except Exception as e:
            print(f"Error loading evaluation records: {e}")

    def toggle_record(self, record_id: int):
        """Toggle whether a record is selected (included) or eliminated."""
        if record_id in self.selected_records:
            del self.selected_records[record_id]
            print(f"     eliminated record {record_id}")
            return False  # Now eliminated
        elif record_id in self.original_records:
            self.selected_records[record_id] = self.original_records[record_id]
            print(f"     restored record {record_id}")
            return True  # Now selected
        return False

    def is_record_selected(self, record_id: int) -> bool:
        """Check if a record is currently selected."""
        return record_id in self.selected_records

    def save_filtered_records(self):
        """Save filtered records with only required fields."""
        if not self.selected_records:
            print("No records to save.")
            return
        
        # Create filtered records with only the required fields
        filtered_records = []
        for record in self.selected_records.values():
            filtered_record = {
                "image_path": record["image_path"],
                "image_path_original": record["image_path_original"],
                "patch_coords": record["patch_coords"]
            }
            filtered_records.append(filtered_record)
        
        # Create output structure
        output_data = {
            "total_records": len(filtered_records),
            "records": filtered_records
        }
        
        # Save to new JSON file
        output_path = self.out_dir / f"filtered_{self.evaluation_records_path.name}"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved {len(filtered_records)} filtered records to: {output_path}")

    def get_records_for_image(self, image_path: str):
        """Get all records for a specific image."""
        records = []
        for record in self.original_records.values():
            if record["image_path"] == image_path:
                records.append(record)
        return records

class CropApp:
    PAN_STEP = 50
    GRID_SIZE = 128

    def __init__(self, evaluation_records_path: Path, out_dir: Path, patch_size: int):
        self.out_dir = out_dir
        self.patch_size = patch_size
        self.writer = EvaluationRecordsWriter(evaluation_records_path, out_dir)
        self.current_image_records = {}  # Map from (grid_row, grid_col) to record_id
        self.current_image_path = None
        
        # Get the first image from the records
        first_image_path = self.get_first_image_path()
        if first_image_path:
            self.load_image(Path(first_image_path))

        self.scr_w, self.scr_h = screen_size()
        if hasattr(self, 'img_w') and hasattr(self, 'img_h'):
            self.view_w = min(self.scr_w - 100, self.img_w)
            self.view_h = min(self.scr_h - 120, self.img_h)
        else:
            self.view_w = self.scr_w - 100
            self.view_h = self.scr_h - 120

        self.off_x = 0
        self.off_y = 0
        self.cur_view = None

        self.win = "Evaluation Records Viewer"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        if hasattr(self, 'view_w') and hasattr(self, 'view_h'):
            cv2.resizeWindow(self.win, self.view_w, self.view_h)
        cv2.setMouseCallback(self.win, self.on_mouse)

    def get_first_image_path(self):
        """Get the first image path from the evaluation records."""
        for record in self.writer.original_records.values():
            return record["image_path"]
        return None

    def load_image(self, path: Path):
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Cannot read {path}")
        self.path = path
        self.img = img
        self.img_h, self.img_w = img.shape[:2]
        self.current_image_path = str(path)
        self.current_image_records = {}  # Reset records for new image
        
        # Load evaluation records for this image
        self.load_records_for_image(str(path))
        
        print(f"\nLoaded {path.name}: {self.img_w}x{self.img_h}px")
        print(f"Found {len(self.current_image_records)} evaluation records for this image")

    def load_records_for_image(self, image_path: str):
        """Load evaluation records for the given image."""
        records = self.writer.get_records_for_image(image_path)
        self.current_image_records = {}
        
        for record in records:
            # Extract grid position from patch_coords or calculate from coordinates
            if "grid_row" in record and "grid_col" in record:
                grid_row = record["grid_row"]
                grid_col = record["grid_col"]
            else:
                # Calculate grid position from patch_coords [x1, y1, x2, y1, x2, y2, x1, y2]
                patch_coords = record["patch_coords"]
                x1, y1 = patch_coords[0], patch_coords[1]
                grid_row = y1 // self.GRID_SIZE
                grid_col = x1 // self.GRID_SIZE
            
            self.current_image_records[(grid_row, grid_col)] = record["record_id"]

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
        
        # Highlight evaluation record patches
        for (grid_row, grid_col), record_id in self.current_image_records.items():
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
                
                # Color based on selection status
                if self.writer.is_record_selected(record_id):
                    # Green for selected/included patches
                    color = (0, 255, 0)
                    thickness = 2
                else:
                    # Red for eliminated patches
                    color = (0, 0, 255)
                    thickness = 3
                
                cv2.rectangle(view, (x1, y1), (x2-1, y2-1), color, thickness)
        
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
                
                # Draw preview rectangle 
                # Check if this position has a record
                if (grid_row, grid_col) in self.current_image_records:
                    record_id = self.current_image_records[(grid_row, grid_col)]
                    if self.writer.is_record_selected(record_id):
                        color = (0, 255, 255)  # Yellow for selected patch preview
                    else:
                        color = (255, 0, 255)  # Magenta for eliminated patch preview
                else:
                    color = (128, 128, 128)  # Gray for areas without records
                cv2.rectangle(view, (x1, y1), (x2-1, y2-1), color, 2)
        
        return view

    def handle_click(self, x, y):
        X = self.off_x + x
        Y = self.off_y + y
        
        # Get grid position
        grid_row, grid_col = self.get_grid_position(X, Y)
        
        # Check if there's a record at this position
        if (grid_row, grid_col) in self.current_image_records:
            record_id = self.current_image_records[(grid_row, grid_col)]
            selected = self.writer.toggle_record(record_id)
            
            if selected:
                print(f"\nRestored record {record_id} at grid[{grid_row}, {grid_col}]")
            else:
                print(f"\nEliminated record {record_id} at grid[{grid_row}, {grid_col}]")
        else:
            print(f"\nNo evaluation record found at grid[{grid_row}, {grid_col}]")

    def run(self):
        print("\nControls:")
        print("  left-click - toggle patch selection (green=selected, red=eliminated)")
        print("  arrow / W/A/X/D - pan")
        print("  o            - open another evaluation records file")
        print("  s            - save filtered records")
        print("  q / ESC      - quit\n")

        while True:
            _, _, win_w, win_h = cv2.getWindowImageRect(self.win)
            self.view_w = min(win_w, self.img_w)
            self.view_h = min(win_h, self.img_h)
            cv2.imshow(self.win, self.current_view())
            selected_count = len(self.writer.selected_records)
            total_count = len(self.writer.original_records)
            cv2.setWindowTitle(
                self.win,
                f"{self.path.name}  [{self.off_x}:{self.off_x+self.view_w},"
                f" {self.off_y}:{self.off_y+self.view_h}] - Selected: {selected_count}/{total_count}")

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                self.writer.save_filtered_records()
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
                    # Load new evaluation records file
                    self.writer = EvaluationRecordsWriter(new_path, self.out_dir)
                    first_image_path = self.get_first_image_path()
                    if first_image_path:
                        self.load_image(Path(first_image_path))
                        self.off_x = self.off_y = 0

        # Save filtered records before closing
        self.writer.save_filtered_records()
        cv2.destroyAllWindows()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-j", "--json",
                   help="Evaluation records JSON file to start with (leave empty for open-file dialog)")
    p.add_argument("-o", "--output_dir", required=True,
                   help="Directory to save filtered records")
    p.add_argument("-s", "--patch_size", type=int, default=128,
                   help="Square patch side (pixels, default: 128)")
    args = p.parse_args()

    if args.json:
        json_file = Path(args.json)
        if not json_file.exists():
            sys.exit(f"  {json_file} not found")
    else:
        json_file = ask_file()
        if json_file is None:
            sys.exit("No evaluation records JSON chosen, bye!")

    app = CropApp(json_file, Path(args.output_dir), args.patch_size)
    app.run()

if __name__ == "__main__":
    main()
