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
        """Load evaluation records from JSON file and filter for FP status only."""
        try:
            with open(self.evaluation_records_path, 'r') as f:
                data = json.load(f)
                if "records" in data:
                    fp_count = 0
                    for record in data["records"]:
                        # Only include records with FP status
                        if record.get("status") == "FP":
                            record_id = record["record_id"]
                            self.original_records[record_id] = record
                            self.selected_records[record_id] = record  # Initially all are selected
                            fp_count += 1
                    
                    print(f"Loaded {fp_count} FP evaluation records out of {len(data['records'])} total records")
                    
                    # Debug: Show first few records to confirm sorting order
                    if fp_count > 0:
                        print(f"\n📊 First 5 FP records (showing anomaly_pixels order):")
                        fp_records_list = list(self.original_records.values())
                        for i, record in enumerate(fp_records_list[:5]):
                            anomaly_pixels = record.get("anomaly_pixels", "N/A")
                            print(f"   {i+1}. Record ID {record['record_id']}: anomaly_pixels = {anomaly_pixels}")
                        
                        if fp_count > 5:
                            print(f"   ... and {fp_count - 5} more records")
                        
                        # Confirm sorting is maintained
                        anomaly_values = [r.get("anomaly_pixels", 0) for r in fp_records_list]
                        if len(anomaly_values) > 1:
                            is_sorted = all(anomaly_values[i] >= anomaly_values[i+1] for i in range(len(anomaly_values)-1))
                            print(f"✅ Records are {'correctly sorted' if is_sorted else 'NOT sorted'} by anomaly_pixels (largest to smallest)")
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
        """Save all records (both selected and deselected) with different structures."""
        if not self.original_records:
            print("No records to save.")
            return
        
        # Create records for both selected and deselected
        all_records = []
        
        for record in self.original_records.values():
            record_id = record["record_id"]
            is_selected = self.is_record_selected(record_id)
            
            if is_selected:
                # Selected records get "normal" structure
                filtered_record = {
                    "image_path": record["image_path"],
                    "image_path_original": record["image_path_original"],
                    "patch_coords": record["patch_coords"],
                    "object": "pcb",
                    "split": "train",
                    "label": "normal",
                    "mask_path": "",
                    "category": "good"
                }
            else:
                # Deselected records get "anomaly" structure
                filtered_record = {
                    "image_path": record["image_path"],
                    "image_path_original": record["image_path_original"],
                    "patch_coords": record["patch_coords"],
                    "object": "pcb",
                    "split": "test",
                    "label": "anomaly",
                    "mask": "",
                    "category": ""
                }
            
            all_records.append(filtered_record)
        
        # Create output structure
        output_data = {
            #"total_records": len(filtered_records),
            "records": all_records
        }
        
        # Save to new JSON file
        output_path = self.out_dir / f"filtered_records_{self.evaluation_records_path.name}"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        selected_count = len(self.selected_records)
        total_count = len(self.original_records)
        print(f"Saved {total_count} total records to: {output_path}")
        print(f"  - {selected_count} selected (normal)")
        print(f"  - {total_count - selected_count} deselected (anomaly)")

    def get_records_for_image(self, image_path: str):
        """Get all records for a specific image."""
        records = []
        for record in self.original_records.values():
            if record["image_path"] == image_path:
                records.append(record)
        return records

class PatchGridApp:
    COLUMNS = 6  # Changed from 6 to 2 columns
    PATCH_SIZE = 512  # Changed from 128 to 256 for twice the size
    PADDING = 10
    SCROLL_STEP = 100

    def __init__(self, evaluation_records_path: Path, out_dir: Path):
        self.out_dir = out_dir
        self.writer = EvaluationRecordsWriter(evaluation_records_path, out_dir)
        
        # Get all FP records (each record represents one patch)
        self.fp_records = list(self.writer.original_records.values())
        
        # Image cache to avoid reloading the same images
        self.image_cache = {}
        
        # Pre-extract all patches to avoid repeated processing
        self.extracted_patches = {}
        self.extract_all_patches()
        
        # Calculate grid layout
        self.calculate_grid_layout()
        
        # Scrolling
        self.scroll_offset = 0
        
        # Label visibility toggle
        self.show_labels = True
        
        # Full info toggle (full path + coordinates vs abbreviated)
        self.show_full_info = False
        
        # Screen setup
        self.scr_w, self.scr_h = screen_size()
        self.view_w = min(self.scr_w - 100, self.total_grid_w)
        self.view_h = min(self.scr_h - 120, self.total_grid_h)
        
        # Window setup
        self.win = "FP Patches Grid Viewer"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.view_w, self.view_h)
        cv2.setMouseCallback(self.win, self.on_mouse)

    def extract_all_patches(self):
        """Pre-extract all patches to avoid repeated processing during display."""
        print("Pre-extracting patches...")
        total_patches = len(self.fp_records)
        
        for i, record in enumerate(self.fp_records):
            if i % 10 == 0:  # Progress indicator
                print(f"Processing patch {i+1}/{total_patches}")
            
            patch = self.extract_patch_from_image(record["image_path"], record["patch_coords"])
            if patch is not None:
                self.extracted_patches[record["record_id"]] = patch
            else:
                # Create a placeholder patch
                placeholder = np.zeros((self.PATCH_SIZE, self.PATCH_SIZE, 3), dtype=np.uint8)
                placeholder.fill(100)  # Gray
                cv2.putText(placeholder, "Error", (50, 128), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self.extracted_patches[record["record_id"]] = placeholder
        
        print(f"Extracted {len(self.extracted_patches)} patches successfully")

    def calculate_grid_layout(self):
        """Calculate the grid layout dimensions."""
        num_patches = len(self.fp_records)
        rows = (num_patches + self.COLUMNS - 1) // self.COLUMNS  # Ceiling division
        
        # Calculate total grid dimensions
        self.total_grid_w = self.COLUMNS * (self.PATCH_SIZE + self.PADDING) + self.PADDING
        self.total_grid_h = rows * (self.PATCH_SIZE + self.PADDING) + self.PADDING
        
        print(f"Grid layout: {rows} rows × {self.COLUMNS} columns = {num_patches} patches")
        print(f"Total grid size: {self.total_grid_w} × {self.total_grid_h} pixels")

    def calculate_line_breaking_threshold(self):
        """Calculate the optimal line breaking threshold based on PATCH_SIZE."""
        # Use proportional buffer instead of fixed 20px
        # Buffer is 10% of PATCH_SIZE (minimum 15px, maximum 50px)
        buffer_pixels = max(3, int(self.PATCH_SIZE * 0.1))
        
        # Estimate characters that can fit in the patch width
        # Assuming each character is roughly 8-10 pixels wide
        # Use proportional buffer for padding and readability
        chars_per_line = (self.PATCH_SIZE - buffer_pixels) // 6

        return chars_per_line, buffer_pixels

    def show_buffer_calculation_details(self):
        """Show detailed buffer calculation information for debugging."""
        chars_per_line, buffer_pixels = self.calculate_line_breaking_threshold()
        buffer_percentage = (buffer_pixels / self.PATCH_SIZE) * 100
        
        print(f"\n📊 Buffer Calculation Details:")
        print(f"   PATCH_SIZE: {self.PATCH_SIZE}px")
        print(f"   Raw buffer (10%): {self.PATCH_SIZE * 0.1:.1f}px")
        print(f"   Applied buffer: {buffer_pixels}px ({buffer_percentage:.1f}%)")
        print(f"   Available text width: {self.PATCH_SIZE - buffer_pixels}px")
        print(f"   Characters per line: {chars_per_line}")
        print(f"   Buffer constraints: min=15px, max=50px")
        
        # Show examples for different patch sizes
        print(f"\n📊 Examples for different PATCH_SIZE values:")
        for test_size in [128, 256, 512, 1024]:
            test_buffer = max(3, int(test_size * 0.1))
            test_chars = (test_size - test_buffer) // 6
            test_percent = (test_buffer / test_size) * 100
            print(f"   PATCH_SIZE {test_size:4d}px → Buffer {test_buffer:3d}px ({test_percent:5.1f}%) → {test_chars:3d} chars/line")

    def get_patch_at_position(self, x, y):
        """Get the patch record at the given screen position."""
        # Adjust for scroll offset
        y += self.scroll_offset
        
        # Calculate grid position
        col = x // (self.PATCH_SIZE + self.PADDING)
        row = y // (self.PATCH_SIZE + self.PADDING)
        
        # Check bounds
        if col < 0 or col >= self.COLUMNS:
            return None
        
        patch_index = row * self.COLUMNS + col
        if patch_index >= len(self.fp_records):
            return None
        
        return patch_index

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_click(x, y)

    def handle_click(self, x, y):
        patch_index = self.get_patch_at_position(x, y)
        if patch_index is not None:
            record = self.fp_records[patch_index]
            record_id = record["record_id"]
            self.writer.toggle_record(record_id)
            
            status = "selected" if self.writer.is_record_selected(record_id) else "eliminated"
            print(f"Toggled record {record_id} to {status}")

    def extract_patch_from_image(self, image_path, patch_coords):
        """Extract a patch from an image based on patch coordinates."""
        try:
            # Check if image is already cached
            if image_path not in self.image_cache:
                img = cv2.imread(image_path)
                if img is None:
                    return None
                self.image_cache[image_path] = img
            else:
                img = self.image_cache[image_path]
            
            # patch_coords format: [x1, y1, x2, y1, x2, y2, x1, y2]
            # Extract the bounding box coordinates
            x1, y1 = patch_coords[0], patch_coords[1]
            x2, y2 = patch_coords[2], patch_coords[5]  # Use x2 and y2 from the coordinates
            
            # Ensure coordinates are within image bounds
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Extract patch
            patch = img[y1:y2, x1:x2]
            
            # Resize to standard patch size if needed
            if patch.shape[:2] != (self.PATCH_SIZE, self.PATCH_SIZE):
                patch = cv2.resize(patch, (self.PATCH_SIZE, self.PATCH_SIZE))
            
            return patch
            
        except Exception as e:
            print(f"Error extracting patch from {image_path}: {e}")
            return None

    def create_grid_view(self):
        """Create the grid view with all patches using pre-extracted data."""
        # Create canvas
        canvas = np.zeros((self.total_grid_h, self.total_grid_w, 3), dtype=np.uint8)
        canvas.fill(50)  # Dark gray background
        
        # Draw patches
        for i, record in enumerate(self.fp_records):
            row = i // self.COLUMNS
            col = i % self.COLUMNS
            
            # Calculate position
            x = col * (self.PATCH_SIZE + self.PADDING) + self.PADDING
            y = row * (self.PATCH_SIZE + self.PADDING) + self.PADDING
            
            # Get pre-extracted patch
            record_id = record["record_id"]
            if record_id in self.extracted_patches:
                patch = self.extracted_patches[record_id]
                
                # Place in grid
                y_start = y - self.scroll_offset
                y_end = y_start + self.PATCH_SIZE
                
                # Only draw if visible
                if y_end > 0 and y_start < self.view_h:
                    y_canvas_start = max(0, y_start)
                    y_canvas_end = min(self.view_h, y_end)
                    y_img_start = max(0, -y_start)
                    y_img_end = y_img_start + (y_canvas_end - y_canvas_start)
                    
                    # Ensure we don't exceed patch boundaries
                    y_img_end = min(y_img_end, patch.shape[0])
                    y_canvas_end = min(y_canvas_end, y_canvas_start + (y_img_end - y_img_start))
                    
                    if y_canvas_end > y_canvas_start and y_img_end > y_img_start:
                        canvas[y_canvas_start:y_canvas_end, x:x+self.PATCH_SIZE] = \
                            patch[y_img_start:y_img_end, :]
                
                # Draw border based on selection status
                if self.writer.is_record_selected(record_id):
                    color = (0, 255, 0)  # Green for selected
                    thickness = 2
                else:
                    color = (0, 0, 255)  # Red for eliminated
                    thickness = 3
                
                # Draw border (only if visible)
                if y_end > 0 and y_start < self.view_h:
                    y_border_start = max(0, y_start)
                    y_border_end = min(self.view_h, y_end)
                    cv2.rectangle(canvas, 
                                (x, y_border_start), 
                                (x + self.PATCH_SIZE, y_border_end - 1), 
                                color, thickness)
                
                # Add patch info label
                if y_end > 0 and y_start < self.view_h and self.show_labels:
                    if self.show_full_info:
                        # Show full absolute path and 8 coordinate values
                        full_path = record["image_path"]
                        coords = record.get("patch_coords", [])
                        if len(coords) == 8:
                            coord_str = f"[{coords[0]},{coords[1]},{coords[2]},{coords[3]},{coords[4]},{coords[5]},{coords[6]},{coords[7]}]"
                        else:
                            coord_str = str(coords)
                        
                        # Get dynamic line breaking threshold based on PATCH_SIZE
                        chars_per_line, buffer_pixels = self.calculate_line_breaking_threshold()
                        
                        # Split long path into multiple lines
                        path_lines = []
                        if len(full_path) > chars_per_line:
                            # Split path into chunks based on available space
                            for i in range(0, len(full_path), chars_per_line):
                                path_lines.append(full_path[i:i+chars_per_line])
                        else:
                            path_lines = [full_path]
                        
                        # Display path lines
                        for line_idx, path_line in enumerate(path_lines):
                            text_y = max(0, y_start) + 15 + (line_idx * 15)
                            if text_y < self.view_h:
                                label = f"ID:{record['record_id']} {path_line}" if line_idx == 0 else f"     {path_line}"
                                cv2.putText(canvas, label, (x + 2, text_y), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                        
                        # Display coordinates on separate line
                        coord_y = max(0, y_start) + 15 + (len(path_lines) * 15)
                        if coord_y < self.view_h:
                            cv2.putText(canvas, f"Coords: {coord_str}", (x + 2, coord_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)
                    else:
                        # Show abbreviated information (original behavior)
                        img_name = Path(record["image_path"]).name
                        if len(img_name) > 12:
                            img_name = img_name[:9] + "..."
                        
                        anomaly_pixels = record.get("anomaly_pixels", "N/A")
                        # Format anomaly_pixels for better readability
                        if isinstance(anomaly_pixels, (int, float)):
                            if anomaly_pixels >= 10000:
                                anomaly_str = f"{anomaly_pixels/1000:.1f}k"
                            else:
                                anomaly_str = str(anomaly_pixels)
                        else:
                            anomaly_str = str(anomaly_pixels)
                        
                        label = f"ID:{record['record_id']} {img_name} ({anomaly_str})"
                        
                        # Calculate text position
                        text_y = max(0, y_start) + 15
                        if text_y < self.view_h:
                            cv2.putText(canvas, label, (x + 2, text_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return canvas

    def run(self):
        print("\nControls:")
        print("  left-click - toggle patch selection (green=selected, red=eliminated)")
        print("  W          - scroll up")
        print("  X          - scroll down")
        print("  h          - toggle label visibility (show/hide patch info)")
        print("  f          - toggle info detail (abbreviated/full path + coordinates)")
        print("  b          - show buffer calculation details")
        print("  o          - open another evaluation records file")
        print("  s          - save filtered records")
        print("  q / ESC    - quit\n")

        # Show initial sorting information
        if self.fp_records:
            print(f"📊 Grid Layout: {len(self.fp_records)} FP patches in 2 columns")
            print(f"📊 Sorting: Records are displayed in order of anomaly_pixels (largest to smallest)")
            print(f"📊 First patch (top-left): anomaly_pixels = {self.fp_records[0].get('anomaly_pixels', 'N/A')}")
            print(f"📊 Last patch (bottom-right): anomaly_pixels = {self.fp_records[-1].get('anomaly_pixels', 'N/A')}")
            print()

        while True:
            try:
                # Create view
                view = self.create_grid_view()
                
                # Crop to view size
                view = view[:self.view_h, :self.view_w]
                
                # Display
                cv2.imshow(self.win, view)
                
                # Update window title
                selected_count = len(self.writer.selected_records)
                total_count = len(self.writer.original_records)
                label_status = "Labels:ON" if self.show_labels else "Labels:OFF"
                info_status = "Info:FULL" if self.show_full_info else "Info:BRIEF"
                cv2.setWindowTitle(
                    self.win,
                    f"FP Patches Grid - Selected: {selected_count}/{total_count} - "
                    f"Scroll: {self.scroll_offset} - {label_status} - {info_status}")

                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('s'):
                    self.writer.save_filtered_records()
                elif key == ord('w'):
                    self.scroll_offset = max(0, self.scroll_offset - self.SCROLL_STEP)
                elif key == ord('x'):
                    max_scroll = max(0, self.total_grid_h - self.view_h)
                    self.scroll_offset = min(max_scroll, self.scroll_offset + self.SCROLL_STEP)
                elif key == ord('h'):
                    self.show_labels = not self.show_labels
                    status = "ON" if self.show_labels else "OFF"
                    print(f"🔍 Label visibility toggled: {status}")
                    print(f"   - Patch info labels are now {status.lower()}")
                    print(f"   - Press 'h' again to toggle back")
                elif key == ord('f'):
                    self.show_full_info = not self.show_full_info
                    status = "ON" if self.show_full_info else "OFF"
                    print(f"🔍 Info detail toggled: {status}")
                    if self.show_full_info:
                        # Calculate and show the line breaking threshold
                        chars_per_line, buffer_pixels = self.calculate_line_breaking_threshold()
                        print(f"   - Now showing: Full absolute path + 8 coordinate values")
                        print(f"   - Full path is displayed with line breaks if needed")
                        print(f"   - Line breaking threshold: {chars_per_line} characters")
                        print(f"   - Buffer: {buffer_pixels}px ({buffer_pixels/self.PATCH_SIZE*100:.1f}% of PATCH_SIZE: {self.PATCH_SIZE})")
                        print(f"   - Example: ID:0 C:\\Users\\...\\image.png")
                        print(f"   - Coords: [x1,y1,x2,y2,x3,y3,x4,y4]")
                    else:
                        print(f"   - Now showing: Abbreviated info (filename + anomaly_pixels)")
                        print(f"   - Example: ID:0 image.png (15.6k)")
                    print(f"   - Press 'f' again to toggle back")
                elif key == ord('b'):
                    self.show_buffer_calculation_details()
                elif key in (ord('o'), ord('O')):
                    new_path = ask_file()
                    if new_path:
                        # Load new evaluation records file
                        self.writer = EvaluationRecordsWriter(new_path, self.out_dir)
                        self.fp_records = list(self.writer.original_records.values())
                        # Clear caches and re-extract patches
                        self.image_cache.clear()
                        self.extracted_patches.clear()
                        self.extract_all_patches()
                        self.calculate_grid_layout()
                        self.scroll_offset = 0
                        
            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                break

        # Save filtered records before closing
        try:
            self.writer.save_filtered_records()
        except Exception as e:
            print(f"Error saving records: {e}")
        cv2.destroyAllWindows()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-j", "--json",
                   help="Evaluation records JSON file to start with (leave empty for open-file dialog)")
    p.add_argument("-o", "--output_dir", required=True,
                   help="Directory to save filtered records")
    args = p.parse_args()

    if args.json:
        json_file = Path(args.json)
        if not json_file.exists():
            sys.exit(f"  {json_file} not found")
    else:
        json_file = ask_file()
        if json_file is None:
            sys.exit("No evaluation records JSON chosen, bye!")

    app = PatchGridApp(json_file, Path(args.output_dir))
    app.run()

if __name__ == "__main__":
    main()
