import os
import re
import torch
import numpy as np

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