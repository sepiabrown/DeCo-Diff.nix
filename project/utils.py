import os
import re
import torch
import numpy as np
from typing import Dict, Set
from PIL import Image as PILImage

def path_to_safe_filename(file_path: str) -> str:
    """
    Convert an absolute file path to a safe filename by replacing path separators with underscores.
    Handles Windows drive letters and both types of path separators.
    Also replaces any file extension with __{extension} format.
    """
    normalized_path = os.path.normpath(file_path)
    # Replace Windows drive letter at the start (e.g., C:\ or C:/) with C__
    normalized_path = re.sub(r'^([a-zA-Z]):[\\/]', r'\1__', normalized_path)
    # Replace all remaining path separators with double underscores
    safe_name = re.sub(r'[\\/]', '__', normalized_path)
    # Replace any file extension with __{extension} format
    safe_name = re.sub(r'\.([a-zA-Z0-9]+)$', r'__\1', safe_name, flags=re.IGNORECASE)
    return safe_name

def safe_filename_to_path(safe_filename: str) -> str:
    """
    Convert a safe filename back to an absolute file path.
    This is the inverse of path_to_safe_filename.
    
    Args:
        safe_filename: Safe filename (e.g., "C__Users__Public__Documents__file__png")
        
    Returns:
        Original file path (e.g., "C:\\Users\\Public\\Documents\\file.png")
    """
    # Replace __{extension} format back to .{extension}
    path = re.sub(r'__([a-zA-Z0-9]+)$', r'.\1', safe_filename, flags=re.IGNORECASE)
    # Replace double underscores with path separators (use string replacement to avoid regex issues)
    path = path.replace('__', os.sep)
    # Replace drive letter format back to Windows format
    # Use string replacement for drive letters to avoid regex escape issues
    for drive_letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        if path.startswith(f'{drive_letter}{os.sep}'):
            path = path.replace(f'{drive_letter}{os.sep}', f'{drive_letter}:{os.sep}', 1)
            break
    
    # Normalize the path to ensure consistent separators
    return os.path.normpath(path)

def _binary_mask(diff: torch.Tensor, threshold: int = 5) -> torch.Tensor:
    """Return a binary mask in ``{0, 1}`` based on *absolute* diff magnitude."""
    return (diff.abs() > (threshold / 255.0)).float()

def _to_numpy(
    t: torch.Tensor,
) -> np.ndarray:  # keep Images API compatibility
    """Detach, move to CPU and convert to ``numpy`` if ``t`` is a tensor."""
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

def _binary_mask_exclude_boundary(diff: torch.Tensor, threshold: int = 5) -> torch.Tensor:
    """
    Return a binary mask in ``{0, 1}`` based on *absolute* diff magnitude,
    but exclude pixels that are adjacent to the boundary of the image.
    
    Args:
        diff: Input tensor with shape (H, W) or (1, H, W)
        threshold: Threshold value for binary conversion (0-255, default: 5)
        
    Returns:
        Binary tensor with same shape as input where boundary-adjacent pixels are excluded
    """
    import cv2
    import numpy as np
    
    # Create initial binary mask
    binary_mask = _binary_mask(diff, threshold)
    
    # Convert to numpy for OpenCV operations
    if binary_mask.dim() == 4 and binary_mask.shape[0] == 1 and binary_mask.shape[1] == 1:
        binary_np = binary_mask.squeeze(0).squeeze(0).cpu().numpy()
    elif binary_mask.dim() == 3 and binary_mask.shape[0] == 1:
        binary_np = binary_mask.squeeze(0).cpu().numpy()
    else:
        binary_np = binary_mask.cpu().numpy()
    
    # Ensure binary values (0 or 1)
    binary_np = (binary_np > 0).astype(np.uint8)
    
    # If the mask is all zeros, return as is
    if np.all(binary_np == 0):
        return binary_mask
    
    # Ensure we have a 2D array
    if binary_np.ndim != 2:
        print(f"Warning: Expected 2D array, got shape {binary_np.shape}")
        return binary_mask
    
    # Create a mask for boundary pixels
    h, w = binary_np.shape
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Mark boundary pixels (first and last row/column)
    boundary_mask[0, :] = 1      # Top row
    boundary_mask[-1, :] = 1     # Bottom row
    boundary_mask[:, 0] = 1      # Left column
    boundary_mask[:, -1] = 1     # Right column
    
    # Find contours of the binary mask
    contours, _ = cv2.findContours(binary_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for pixels adjacent to boundary
    adjacent_to_boundary = np.zeros_like(binary_np)
    
    for contour in contours:
        # Create a mask for this contour
        contour_mask = np.zeros_like(binary_np)
        cv2.fillPoly(contour_mask, [contour], (1,))
        
        # Check if this contour touches the boundary
        if np.any(contour_mask * boundary_mask):
            # This contour touches the boundary, mark all pixels in this contour as adjacent to boundary
            adjacent_to_boundary = np.logical_or(adjacent_to_boundary, contour_mask)
    
    # Remove pixels adjacent to boundary from the original binary mask
    result_np = binary_np * (1 - adjacent_to_boundary)
    
    # Convert back to tensor
    result_tensor = torch.from_numpy(result_np).float()
    
    # Move to the same device as input
    if diff.is_cuda:
        result_tensor = result_tensor.cuda()
    
    # Restore original tensor shape
    if diff.dim() == 4:
        result_tensor = result_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    elif diff.dim() == 3:
        result_tensor = result_tensor.unsqueeze(0)  # Shape: (1, H, W)
    # If diff.dim() == 2, keep as is (H, W)
    
    return result_tensor

def _binary_mask_exclude_boundary2(diff: torch.Tensor, threshold: int = 5, debug: bool = False, visualize: bool = False, save_path: str = None, filename: str = None) -> torch.Tensor:
    """
    Return a binary mask in ``{0, 1}`` based on *absolute* diff magnitude,
    but exclude pixels that are adjacent to the boundary of the image,
    and also exclude small connected components (less than 6 pixels) within 12 pixels of boundary.
    
    Args:
        diff: Input tensor with shape (H, W) or (1, H, W)
        threshold: Threshold value for binary conversion (0-255, default: 5)
        debug: Enable debug output (default: False)
        visualize: Enable visualization of the process (default: False)
        save_path: Path to save visualization image (default: None, saves to current directory)
        filename: Filename to check for "108826_198" pattern (default: None)
        
    Returns:
        Binary tensor with same shape as input where boundary-adjacent pixels and small boundary components are excluded
    """
    
    # Check if filename contains "108826_198" for debug and visualization
    should_debug = debug and filename and "108826_198" in filename
    should_visualize = visualize and filename and "108826_198" in filename
    import cv2
    import numpy as np
    
    # Create initial binary mask
    binary_mask = _binary_mask(diff, threshold)
    
    # Convert to numpy for OpenCV operations
    if binary_mask.dim() == 4 and binary_mask.shape[0] == 1 and binary_mask.shape[1] == 1:
        binary_np = binary_mask.squeeze(0).squeeze(0).cpu().numpy()
    elif binary_mask.dim() == 3 and binary_mask.shape[0] == 1:
        binary_np = binary_mask.squeeze(0).cpu().numpy()
    else:
        binary_np = binary_mask.cpu().numpy()
    
    # Ensure binary values (0 or 1)
    binary_np = (binary_np > 0).astype(np.uint8)
    
    # If the mask is all zeros, return as is
    if np.all(binary_np == 0):
        return binary_mask
    
    # Ensure we have a 2D array
    if binary_np.ndim != 2:
        print(f"Warning: Expected 2D array, got shape {binary_np.shape}")
        return binary_mask
    
    # Create a mask for boundary pixels
    h, w = binary_np.shape
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Mark boundary pixels (first and last row/column)
    boundary_mask[0, :] = 1      # Top row
    boundary_mask[-1, :] = 1     # Bottom row
    boundary_mask[:, 0] = 1      # Left column
    boundary_mask[:, -1] = 1     # Right column
    
    # Find contours of the binary mask
    contours, _ = cv2.findContours(binary_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for pixels to exclude
    pixels_to_exclude = np.zeros_like(binary_np)
    
    # Create a distance map from boundary (12 pixels inward)
    distance_from_boundary = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate distance from boundary for each pixel
    for i in range(h):
        for j in range(w):
            # Distance from top, bottom, left, right boundaries
            dist_top = i
            dist_bottom = h - 1 - i
            dist_left = j
            dist_right = w - 1 - j
            # Minimum distance from any boundary
            min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
            distance_from_boundary[i, j] = min_dist
    
    for contour in contours:
        # Create a mask for this contour
        contour_mask = np.zeros_like(binary_np)
        cv2.fillPoly(contour_mask, [contour], (1,))
        
        # Check if this contour touches the boundary
        if np.any(contour_mask * boundary_mask):
            # This contour touches the boundary, mark all pixels in this contour as adjacent to boundary
            pixels_to_exclude = np.logical_or(pixels_to_exclude, contour_mask)
        else:
            # Check if this contour is within 12 pixels of boundary and has less than 6 pixels
            contour_pixels = np.sum(contour_mask)
            if contour_pixels < 6:
                # Check if any pixel in this contour is within 12 pixels of boundary
                contour_within_12 = np.logical_and(contour_mask, distance_from_boundary < 12)
                if np.any(contour_within_12):
                    # This small contour is within 12 pixels of boundary, exclude it
                    pixels_to_exclude = np.logical_or(pixels_to_exclude, contour_mask)
    
    # Additional step: Remove individual anomaly pixels that are within 12 pixels of boundary and have less than 6 pixels
    # Find connected components in the remaining binary mask
    remaining_binary = binary_np * (1 - pixels_to_exclude)
    
    # Ensure remaining_binary is uint8 for OpenCV
    remaining_binary = remaining_binary.astype(np.uint8)
    
    # Use connected components analysis to find individual pixels and small groups
    num_labels, labels = cv2.connectedComponents(remaining_binary, connectivity=8)
    
    if should_debug:
        print(f"Found {num_labels-1} connected components in remaining binary")
    
    # Check each connected component
    for label in range(1, num_labels):  # Skip background (label 0)
        # Create mask for this component
        component_mask = (labels == label).astype(np.uint8)
        component_pixels = np.sum(component_mask)
        
        if should_debug and component_pixels < 6:
            # Find the position of this component
            component_positions = np.where(component_mask > 0)
            min_distances = []
            for i, j in zip(component_positions[0], component_positions[1]):
                min_distances.append(distance_from_boundary[i, j])
            min_distance = min(min_distances) if min_distances else 999
            
            print(f"Component {label}: {component_pixels} pixels, min distance from boundary: {min_distance}")
        
        # If component has less than 6 pixels and is within 12 pixels of boundary
        if component_pixels < 6:
            # Check if any pixel in this component is within 12 pixels of boundary
            component_within_12 = np.logical_and(component_mask, distance_from_boundary <= 12)
            if np.any(component_within_12):
                # This small component is within 12 pixels of boundary, exclude it
                pixels_to_exclude = np.logical_or(pixels_to_exclude, component_mask)
                if should_debug:
                    print(f"  -> Excluded component {label} ({component_pixels} pixels)")
            elif should_debug:
                print(f"  -> Component {label} NOT excluded (distance >= 12)")
    
    # Remove excluded pixels from the original binary mask
    result_np = binary_np * (1 - pixels_to_exclude)
    
    if should_debug:
        original_pixels = np.sum(binary_np)
        excluded_pixels = np.sum(pixels_to_exclude)
        remaining_pixels = np.sum(result_np)
        print(f"Original pixels: {original_pixels}, Excluded: {excluded_pixels}, Remaining: {remaining_pixels}")
        
        # Check for any remaining 1-pixel components
        remaining_labels, remaining_labels_array = cv2.connectedComponents(result_np.astype(np.uint8), connectivity=8)
        single_pixels = 0
        for label in range(1, remaining_labels):
            component_mask = (remaining_labels_array == label).astype(np.uint8)
            if np.sum(component_mask) == 1:
                single_pixels += 1
        print(f"Remaining single pixels: {single_pixels}")
    
    # Convert back to tensor
    result_tensor = torch.from_numpy(result_np).float()
    
    # Move to the same device as input
    if diff.is_cuda:
        result_tensor = result_tensor.cuda()
    
    # Restore original tensor shape
    if diff.dim() == 4:
        result_tensor = result_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    elif diff.dim() == 3:
        result_tensor = result_tensor.unsqueeze(0)  # Shape: (1, H, W)
    # If diff.dim() == 2, keep as is (H, W)
    
    # Visualization
    if should_visualize:
        try:
            import matplotlib.pyplot as plt
            
            # Get the original binary mask for comparison
            original_binary = _binary_mask(diff, threshold)
            if original_binary.dim() == 4:
                original_binary_np = original_binary.squeeze(0).squeeze(0).cpu().numpy()
            elif original_binary.dim() == 3:
                original_binary_np = original_binary.squeeze(0).cpu().numpy()
            else:
                original_binary_np = original_binary.cpu().numpy()
            
            # Get the result as numpy array
            if result_tensor.dim() == 4:
                result_np_viz = result_tensor.squeeze(0).squeeze(0).cpu().numpy()
            elif result_tensor.dim() == 3:
                result_np_viz = result_tensor.squeeze(0).cpu().numpy()
            else:
                result_np_viz = result_tensor.cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original binary mask
            axes[0].imshow(original_binary_np, cmap='gray')
            axes[0].set_title(f'Original Binary Mask\n({np.sum(original_binary_np > 0)} pixels)')
            axes[0].axis('off')
            
            # Excluded pixels mask
            excluded_mask = original_binary_np * pixels_to_exclude
            axes[1].imshow(excluded_mask, cmap='hot')
            axes[1].set_title(f'Excluded Pixels\n({np.sum(excluded_mask > 0)} pixels)')
            axes[1].axis('off')
            
            # Final result
            axes[2].imshow(result_np_viz, cmap='gray')
            axes[2].set_title(f'After Boundary Exclusion\n({np.sum(result_np_viz > 0)} pixels)')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save or show the visualization
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to: {save_path}")
            
            plt.show()
            plt.close()
            
        except ImportError:
            print("Warning: matplotlib not available for visualization")
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
    
    return result_tensor

def _binary_mask_exclude_boundary3(diff: torch.Tensor, threshold: int = 5, debug: bool = False, visualize: bool = False, save_path: str = None, filename: str = None) -> torch.Tensor:
    """
    Return a binary mask in ``{0, 1}`` based on *absolute* diff magnitude,
    but exclude pixels that are adjacent to the boundary of the image,
    and also exclude small connected components (less than 6 pixels) within 12 pixels of boundary.
    
    Args:
        diff: Input tensor with shape (H, W) or (1, H, W)
        threshold: Threshold value for binary conversion (0-255, default: 5)
        debug: Enable debug output (default: False)
        visualize: Enable visualization of the process (default: False)
        save_path: Path to save visualization image (default: None, saves to current directory)
        filename: Filename to check for "108826_198" pattern (default: None)
        
    Returns:
        Binary tensor with same shape as input where boundary-adjacent pixels and small boundary components are excluded
    """
    
    # Check if filename contains "108826_198" for debug and visualization
    should_debug = debug and filename and "108826_198" in filename
    should_visualize = visualize and filename and "108826_198" in filename
    import cv2
    import numpy as np
    
    # Create initial binary mask
    binary_mask = _binary_mask(diff, threshold)
    
    # Convert to numpy for OpenCV operations
    if binary_mask.dim() == 4 and binary_mask.shape[0] == 1 and binary_mask.shape[1] == 1:
        binary_np = binary_mask.squeeze(0).squeeze(0).cpu().numpy()
    elif binary_mask.dim() == 3 and binary_mask.shape[0] == 1:
        binary_np = binary_mask.squeeze(0).cpu().numpy()
    else:
        binary_np = binary_mask.cpu().numpy()
    
    # Ensure binary values (0 or 1)
    binary_np = (binary_np > 0).astype(np.uint8)
    
    # If the mask is all zeros, return as is
    if np.all(binary_np == 0):
        return binary_mask
    
    # Ensure we have a 2D array
    if binary_np.ndim != 2:
        print(f"Warning: Expected 2D array, got shape {binary_np.shape}")
        return binary_mask
    
    # Create a mask for boundary pixels
    h, w = binary_np.shape
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Mark boundary pixels (first and last row/column)
    boundary_mask[0, :] = 1      # Top row
    boundary_mask[-1, :] = 1     # Bottom row
    boundary_mask[:, 0] = 1      # Left column
    boundary_mask[:, -1] = 1     # Right column
    
    # Create a mask for pixels to exclude
    pixels_to_exclude = np.zeros_like(binary_np)
    
    # Create a distance map from boundary (12 pixels inward)
    distance_from_boundary = np.zeros((h, w), dtype=np.uint8)
    
    remaining_binary = binary_np
    
    # Ensure remaining_binary is uint8 for OpenCV
    remaining_binary = remaining_binary.astype(np.uint8)
    
    # Use connected components analysis to find individual pixels and small groups
    num_labels, labels = cv2.connectedComponents(remaining_binary, connectivity=8)
    
    if should_debug:
        print(f"Found {num_labels-1} connected components in remaining binary")
    
    # Check each connected component
    for label in range(1, num_labels):  # Skip background (label 0)
        # Create mask for this component
        component_mask = (labels == label).astype(np.uint8)
        component_pixels = np.sum(component_mask)
        
        if should_debug and component_pixels < 10:
            # Find the position of this component
            component_positions = np.where(component_mask > 0)
            min_distances = []
            for i, j in zip(component_positions[0], component_positions[1]):
                min_distances.append(distance_from_boundary[i, j])
            min_distance = min(min_distances) if min_distances else 999
            
            print(f"Component {label}: {component_pixels} pixels, min distance from boundary: {min_distance}")
        
        # If component has less than 6 pixels and is within 12 pixels of boundary
        if component_pixels < 10:
            # Check if any pixel in this component is within 12 pixels of boundary
            pixels_to_exclude = np.logical_or(pixels_to_exclude, component_mask)
            if should_debug:
                print(f"  -> Excluded component {label} ({component_pixels} pixels)")

    # Remove excluded pixels from the original binary mask
    result_np = binary_np * (1 - pixels_to_exclude)
    
    if should_debug:
        original_pixels = np.sum(binary_np)
        excluded_pixels = np.sum(pixels_to_exclude)
        remaining_pixels = np.sum(result_np)
        print(f"Original pixels: {original_pixels}, Excluded: {excluded_pixels}, Remaining: {remaining_pixels}")
        
        # Check for any remaining 1-pixel components
        remaining_labels, remaining_labels_array = cv2.connectedComponents(result_np.astype(np.uint8), connectivity=8)
        single_pixels = 0
        for label in range(1, remaining_labels):
            component_mask = (remaining_labels_array == label).astype(np.uint8)
            if np.sum(component_mask) == 1:
                single_pixels += 1
        print(f"Remaining single pixels: {single_pixels}")
    
    # Convert back to tensor
    result_tensor = torch.from_numpy(result_np).float()
    
    # Move to the same device as input
    if diff.is_cuda:
        result_tensor = result_tensor.cuda()
    
    # Restore original tensor shape
    if diff.dim() == 4:
        result_tensor = result_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    elif diff.dim() == 3:
        result_tensor = result_tensor.unsqueeze(0)  # Shape: (1, H, W)
    # If diff.dim() == 2, keep as is (H, W)
    
    # Visualization
    if should_visualize:
        try:
            import matplotlib.pyplot as plt
            
            # Get the original binary mask for comparison
            original_binary = _binary_mask(diff, threshold)
            if original_binary.dim() == 4:
                original_binary_np = original_binary.squeeze(0).squeeze(0).cpu().numpy()
            elif original_binary.dim() == 3:
                original_binary_np = original_binary.squeeze(0).cpu().numpy()
            else:
                original_binary_np = original_binary.cpu().numpy()
            
            # Get the result as numpy array
            if result_tensor.dim() == 4:
                result_np_viz = result_tensor.squeeze(0).squeeze(0).cpu().numpy()
            elif result_tensor.dim() == 3:
                result_np_viz = result_tensor.squeeze(0).cpu().numpy()
            else:
                result_np_viz = result_tensor.cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original binary mask
            axes[0].imshow(original_binary_np, cmap='gray')
            axes[0].set_title(f'Original Binary Mask\n({np.sum(original_binary_np > 0)} pixels)')
            axes[0].axis('off')
            
            # Excluded pixels mask
            excluded_mask = original_binary_np * pixels_to_exclude
            axes[1].imshow(excluded_mask, cmap='hot')
            axes[1].set_title(f'Excluded Pixels\n({np.sum(excluded_mask > 0)} pixels)')
            axes[1].axis('off')
            
            # Final result
            axes[2].imshow(result_np_viz, cmap='gray')
            axes[2].set_title(f'After Boundary Exclusion\n({np.sum(result_np_viz > 0)} pixels)')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save or show the visualization
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to: {save_path}")
            
            plt.show()
            plt.close()
            
        except ImportError:
            print("Warning: matplotlib not available for visualization")
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
    
    return result_tensor

def load_original_images(image_paths: Set[str]) -> Dict[str, np.ndarray]:
    """
    Load original images for all image paths.
    Handles both safe filenames and actual file paths.
    
    Args:
        image_paths: Set of image paths to load (can be safe filenames or actual paths)
        
    Returns:
        Dictionary mapping image paths to loaded images as numpy arrays
    """
    original_images = {}
    
    for image_path in image_paths:
        try:
            # First try the path as-is
            if os.path.exists(image_path):
                original_image = np.array(PILImage.open(image_path).convert('RGB'))
                original_images[image_path] = original_image
            else:
                # Try converting from safe filename to actual path
                # Only try this if the path looks like a safe filename (contains '__')
                if '__' in image_path:
                    try:
                        actual_path = safe_filename_to_path(image_path)
                        if os.path.exists(actual_path):
                            original_image = np.array(PILImage.open(actual_path).convert('RGB'))
                            original_images[image_path] = original_image  # Keep safe filename as key
                        else:
                            print(f"Warning: Original image not found: {image_path} (tried: {actual_path})")
                    except Exception as e:
                        print(f"Warning: Error converting safe filename {image_path}: {e}")
                else:
                    print(f"Warning@: Original image not found: {image_path}")
        except Exception as e:
            print(f"Warning: Error loading original image {image_path}: {e}")
    
    return original_images