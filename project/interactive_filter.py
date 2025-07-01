import argparse
import os
import sys
from pathlib import Path

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

class PatchWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, original_img: np.ndarray, x, y, size, image_path: Path):
        stem = image_path.stem
        path = (self.out_dir / f"{stem}_{x}_{y}_{size}.png").resolve()
        
        # Create a copy of the original image
        masked_img = original_img.copy()
        
        # Create a black mask for the entire image
        mask = np.zeros_like(original_img)
        
        # Calculate the region to keep (the clicked area)
        half = size // 2
        x1 = np.clip(x - half, 0, original_img.shape[1] - size)
        y1 = np.clip(y - half, 0, original_img.shape[0] - size)
        x2 = x1 + size
        y2 = y1 + size
        
        # Copy the original region to the mask
        mask[y1:y2, x1:x2] = original_img[y1:y2, x1:x2]
        
        # Save the masked image
        cv2.imwrite(str(path), mask)
        print(f"     saved -> {path}")

    def save_cumulative(self, cumulative_mask: np.ndarray, image_path: Path):
        stem = image_path.stem
        path = (self.out_dir / f"{stem}_cumulative.png").resolve()
        
        # Save the cumulative mask
        cv2.imwrite(str(path), cumulative_mask)
        print(f"     saved cumulative -> {path}")
        
        # Apply line detection to extract prominent line segments
        self.extract_line_segments(cumulative_mask, image_path, cumulative_mask)
    
    def extract_line_segments(self, image: np.ndarray, image_path: Path, cumulative_mask: np.ndarray):
        stem = image_path.stem
        
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Save edges
        edges_path = (self.out_dir / f"{stem}_edges.png").resolve()
        cv2.imwrite(str(edges_path), edges)
        print(f"     saved edges -> {edges_path}")
        
        # Apply Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        # Create a black image to draw lines on
        line_image = np.zeros_like(image)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Save line segments
        lines_path = (self.out_dir / f"{stem}_line_segments.png").resolve()
        cv2.imwrite(str(lines_path), line_image)
        print(f"     saved line segments -> {lines_path}")
        
        # Alternative: Use LSD (Line Segment Detector) for more precise detection
        try:
            # Create LSD detector
            lsd = cv2.createLineSegmentDetector(0)
            lines_lsd = lsd.detect(gray)[0]
            
            # Create image for LSD lines
            lsd_image = np.zeros_like(image)
            
            if lines_lsd is not None:
                for line in lines_lsd:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(lsd_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
            
            # Save LSD line segments
            lsd_path = (self.out_dir / f"{stem}_lsd_segments.png").resolve()
            cv2.imwrite(str(lsd_path), lsd_image)
            print(f"     saved LSD segments -> {lsd_path}")
            
        except Exception as e:
            print(f"     LSD detection failed: {e}")
        
        # Create a combined result showing original cumulative + detected lines
        combined = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Save combined result
        combined_path = (self.out_dir / f"{stem}_with_lines.png").resolve()
        cv2.imwrite(str(combined_path), combined)
        print(f"     saved combined -> {combined_path}")
        
        # Scratch detection
        self.detect_scratches(gray, image_path, cumulative_mask)
    
    def detect_scratches(self, gray: np.ndarray, image_path: Path, cumulative_mask: np.ndarray):
        stem = image_path.stem
        
        # Use the original image for scratch detection (don't filter out selected areas)
        gray_for_detection = gray.copy()
        
        # Method 1: Morphological operations for scratch detection
        # Create morphological kernels for scratch detection
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        kernel_diagonal1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        
        # Apply morphological operations on original image
        morph_h = cv2.morphologyEx(gray_for_detection, cv2.MORPH_OPEN, kernel_horizontal)
        morph_v = cv2.morphologyEx(gray_for_detection, cv2.MORPH_OPEN, kernel_vertical)
        morph_d = cv2.morphologyEx(gray_for_detection, cv2.MORPH_OPEN, kernel_diagonal1)
        
        # Combine morphological results
        morph_combined = cv2.bitwise_or(morph_h, morph_v)
        morph_combined = cv2.bitwise_or(morph_combined, morph_d)
        
        # Apply cumulative mask to make non-selected areas black
        if cumulative_mask is not None:
            if len(cumulative_mask.shape) == 3:
                cumulative_gray = cv2.cvtColor(cumulative_mask, cv2.COLOR_BGR2GRAY)
            else:
                cumulative_gray = cumulative_mask.copy()
            morph_combined = cv2.bitwise_and(morph_combined, cumulative_gray)
        
        # Save morphological result
        morph_path = (self.out_dir / f"{stem}_morphological.png").resolve()
        cv2.imwrite(str(morph_path), morph_combined)
        print(f"     saved morphological -> {morph_path}")
        
        # Method 2: Gradient-based scratch detection
        # Calculate gradients on original image
        grad_x = cv2.Sobel(gray_for_detection, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_for_detection, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_dir = np.arctan2(grad_y, grad_x)
        
        # Normalize gradient magnitude
        grad_mag_norm = np.zeros_like(grad_mag)
        cv2.normalize(grad_mag, grad_mag_norm, 0, 255, cv2.NORM_MINMAX)
        grad_mag_norm = grad_mag_norm.astype(np.uint8)
        
        # Threshold gradient magnitude to find strong edges (potential scratches)
        _, grad_thresh = cv2.threshold(grad_mag_norm, 50, 255, cv2.THRESH_BINARY)
        
        # Apply cumulative mask to make non-selected areas black
        if cumulative_mask is not None:
            if len(cumulative_mask.shape) == 3:
                cumulative_gray = cv2.cvtColor(cumulative_mask, cv2.COLOR_BGR2GRAY)
            else:
                cumulative_gray = cumulative_mask.copy()
            grad_thresh = cv2.bitwise_and(grad_thresh, cumulative_gray)
        
        # Save gradient-based result
        grad_path = (self.out_dir / f"{stem}_gradient.png").resolve()
        cv2.imwrite(str(grad_path), grad_thresh)
        print(f"     saved gradient -> {grad_path}")
        
        # Method 3: Local Binary Pattern (LBP) for texture analysis
        # Calculate LBP for texture analysis on original image
        lbp = self.calculate_lbp(gray_for_detection)
        
        # Threshold LBP to find texture irregularities (scratches)
        _, lbp_thresh = cv2.threshold(lbp, 200, 255, cv2.THRESH_BINARY)
        
        # Apply cumulative mask to make non-selected areas black
        if cumulative_mask is not None:
            if len(cumulative_mask.shape) == 3:
                cumulative_gray = cv2.cvtColor(cumulative_mask, cv2.COLOR_BGR2GRAY)
            else:
                cumulative_gray = cumulative_mask.copy()
            lbp_thresh = cv2.bitwise_and(lbp_thresh, cumulative_gray)
        
        # Save LBP result
        lbp_path = (self.out_dir / f"{stem}_lbp.png").resolve()
        cv2.imwrite(str(lbp_path), lbp_thresh)
        print(f"     saved LBP -> {lbp_path}")
        
        # Method 4: Contour-based scratch detection
        # Apply adaptive thresholding on original image
        adaptive_thresh = cv2.adaptiveThreshold(gray_for_detection, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on aspect ratio and area (scratch-like characteristics)
        scratch_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Minimum area threshold reduced from 50 to 10
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)  # Avoid division by zero
                if aspect_ratio > 3:  # Scratches are typically long and thin
                    scratch_contours.append(contour)
        
        # Draw scratch contours
        contour_image = np.zeros_like(gray)
        cv2.drawContours(contour_image, scratch_contours, -1, (255, 255, 255), 2)
        
        # Apply cumulative mask to make non-selected areas black
        if cumulative_mask is not None:
            if len(cumulative_mask.shape) == 3:
                cumulative_gray = cv2.cvtColor(cumulative_mask, cv2.COLOR_BGR2GRAY)
            else:
                cumulative_gray = cumulative_mask.copy()
            contour_image = cv2.bitwise_and(contour_image, cumulative_gray)
        
        # Save contour result
        contour_path = (self.out_dir / f"{stem}_contours.png").resolve()
        cv2.imwrite(str(contour_path), contour_image)
        print(f"     saved contours -> {contour_path}")
        
        # Method 5: Combined scratch detection
        # Combine all methods
        combined_scratch = np.zeros_like(gray)
        
        # Add morphological result
        combined_scratch = cv2.bitwise_or(combined_scratch, morph_combined)
        
        # Add gradient result
        combined_scratch = cv2.bitwise_or(combined_scratch, grad_thresh)
        
        # Add LBP result
        combined_scratch = cv2.bitwise_or(combined_scratch, lbp_thresh)
        
        # Add contour result
        combined_scratch = cv2.bitwise_or(combined_scratch, contour_image)
        
        # Apply morphological closing to connect nearby scratch pixels
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_scratch = cv2.morphologyEx(combined_scratch, cv2.MORPH_CLOSE, kernel_close)
        
        # Save combined scratch detection
        combined_scratch_path = (self.out_dir / f"{stem}_scratches_combined.png").resolve()
        cv2.imwrite(str(combined_scratch_path), combined_scratch)
        print(f"     saved combined scratches -> {combined_scratch_path}")
        
        # Create overlay showing scratches on original
        # Convert gray back to BGR for overlay
        if len(gray.shape) == 2:
            overlay_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            overlay_img = gray.copy()
            
        scratch_overlay = np.zeros_like(overlay_img)
        scratch_overlay[combined_scratch > 0] = [0, 0, 255]  # Red for scratches
        
        # Blend overlay with original image
        alpha = 0.7
        overlay = cv2.addWeighted(overlay_img, 1-alpha, scratch_overlay, alpha, 0)
        
        # Save overlay
        overlay_path = (self.out_dir / f"{stem}_scratches_overlay.png").resolve()
        cv2.imwrite(str(overlay_path), overlay)
        print(f"     saved scratch overlay -> {overlay_path}")
    
    def calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern for texture analysis"""
        height, width = image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = image[i, j]
                code = 0
                
                # 8-neighbor LBP
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp

class CropApp:
    PAN_STEP = 50

    def __init__(self, image_path: Path, out_dir: Path, patch_size: int):
        self.out_dir = out_dir
        self.patch_size = patch_size
        self.writer = PatchWriter(out_dir)
        self.load_image(image_path)

        self.scr_w, self.scr_h = screen_size()
        self.view_w = min(self.scr_w - 100, self.img_w)
        self.view_h = min(self.scr_h - 120, self.img_h)

        self.off_x = 0
        self.off_y = 0
        self.cur_view = None
        
        # Initialize cumulative mask
        self.cumulative_mask = np.zeros_like(self.img)
        
        # Drawing state variables
        self.drawing = False
        self.drawing_points = []
        self.drawing_mode = True  # True for freehand drawing, False for fixed patches

        self.win = "Cropper"
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
        # Reset cumulative mask for new image
        self.cumulative_mask = np.zeros_like(self.img)
        print(f"\nLoaded {path.name}: {self.img_w}x{self.img_h}px")

    def pan(self, dx: int, dy: int):
        self.off_x = np.clip(self.off_x + dx, 0, max(0, self.img_w - self.view_w))
        self.off_y = np.clip(self.off_y + dy, 0, max(0, self.img_h - self.view_h))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.cur_view = (x, y)
            if self.drawing and self.drawing_mode:
                # Add point to drawing path
                abs_x = self.off_x + x
                abs_y = self.off_y + y
                self.drawing_points.append((abs_x, abs_y))
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing_mode:
                # Start drawing
                self.drawing = True
                self.drawing_points = []
                abs_x = self.off_x + x
                abs_y = self.off_y + y
                self.drawing_points.append((abs_x, abs_y))
            else:
                # Fixed patch mode
                self.handle_click(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.drawing_mode:
                # Finish drawing and process the region
                self.drawing = False
                if len(self.drawing_points) > 2:
                    self.handle_drawn_region()

    def handle_drawn_region(self):
        if len(self.drawing_points) < 3:
            return
            
        # Convert points to numpy array for OpenCV
        points = np.array(self.drawing_points, dtype=np.int32)
        
        # Create a mask for the drawn region
        mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], (255,))
        
        # Apply the mask to the original image
        region = cv2.bitwise_and(self.img, self.img, mask=mask)
        
        # Add to cumulative mask
        self.cumulative_mask = cv2.bitwise_or(self.cumulative_mask, region)
        
        # Save the cumulative result
        self.writer.save_cumulative(self.cumulative_mask, self.path)
        print(f"\nDrew region with {len(self.drawing_points)} points")

    def current_view(self):
        view = self.img[self.off_y:self.off_y + self.view_h,
                        self.off_x:self.off_x + self.view_w].copy()
        
        # Draw the current drawing path if in drawing mode
        if self.drawing_mode and self.drawing and len(self.drawing_points) > 1:
            # Convert absolute coordinates to view coordinates
            view_points = []
            for abs_x, abs_y in self.drawing_points:
                view_x = abs_x - self.off_x
                view_y = abs_y - self.off_y
                if 0 <= view_x < self.view_w and 0 <= view_y < self.view_h:
                    view_points.append((view_x, view_y))
            
            if len(view_points) > 1:
                # Draw the current path
                for i in range(1, len(view_points)):
                    cv2.line(view, view_points[i-1], view_points[i], (0, 255, 0), 2)
        
        # Show preview of fixed patch if not in drawing mode
        elif not self.drawing_mode and self.cur_view is not None:
            vx, vy = self.cur_view
            X, Y = self.off_x + vx, self.off_y + vy
            half = self.patch_size // 2
            x1 = np.clip(X - half, 0, self.img_w - self.patch_size)
            y1 = np.clip(Y - half, 0, self.img_h - self.patch_size)
            rx1, ry1 = x1 - self.off_x, y1 - self.off_y
            cv2.rectangle(view, (rx1, ry1),
                          (rx1 + self.patch_size - 1, ry1 + self.patch_size - 1),
                          (0,255,0), max(1, self.patch_size // 75))
        
        return view

    def handle_click(self, x, y):
        X = self.off_x + x
        Y = self.off_y + y

        half = self.patch_size // 2

        x1 = np.clip(X - half, 0, self.img_w - self.patch_size)
        y1 = np.clip(Y - half, 0, self.img_h - self.patch_size)
        x2 = x1 + self.patch_size
        y2 = y1 + self.patch_size
        
        # Add this region to the cumulative mask
        self.cumulative_mask[y1:y2, x1:x2] = self.img[y1:y2, x1:x2]

        view = self.current_view().copy()
        rx1, ry1 = x1 - self.off_x, y1 - self.off_y

        roi = view[ry1:ry1 + self.patch_size, rx1:rx1 + self.patch_size]
        bright_roi = cv2.add(roi, np.full_like(roi, 80))
        roi[:] = bright_roi

        cv2.rectangle(
            view,
            (rx1, ry1),
            (rx1 + self.patch_size - 1, ry1 + self.patch_size - 1),
            (0,0,255),
            (max(2, self.patch_size//50))
        )

        cv2.imshow(self.win, view)
        cv2.waitKey(120)
        # Save the cumulative result
        self.writer.save_cumulative(self.cumulative_mask, self.path)
        print(f"\nClicked at ({x},{y}) -> crop [{x1}:{x2}, {y1}:{y2}] (cumulative)")

    def run(self):
        print("\nControls:")
        print("  left-click and drag - draw custom region (default mode)")
        print("  t                  - toggle between drawing and fixed patch mode")
        print("  arrow / WASD       - pan")
        print("  o                  - open another file")
        print("  q / ESC            - quit\n")

        while True:
            _, _, win_w, win_h = cv2.getWindowImageRect(self.win)
            self.view_w = min(win_w, self.img_w)
            self.view_h = min(win_h, self.img_h)
            cv2.imshow(self.win, self.current_view())
            
            mode_text = "DRAWING" if self.drawing_mode else "FIXED PATCH"
            cv2.setWindowTitle(
                self.win,
                f"{self.path.name} [{mode_text}]  [{self.off_x}:{self.off_x+self.view_w},"
                f" {self.off_y}:{self.off_y+self.view_h}]")

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('w'), 82):
                self.pan(0, -self.PAN_STEP)
            elif key in (ord('s'), 84):
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
            elif key in (ord('t'), ord('T')):
                self.drawing_mode = not self.drawing_mode
                print(f"Switched to {'DRAWING' if self.drawing_mode else 'FIXED PATCH'} mode")

        cv2.destroyAllWindows()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--image",
                   help="Image to start with (leave empty for open-file dialog)")
    p.add_argument("-o", "--output_dir", required=True,
                   help="Directory to save patches")
    p.add_argument("-s", "--patch_size", type=int, required=True,
                   help="Square patch side (pixels)")
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
