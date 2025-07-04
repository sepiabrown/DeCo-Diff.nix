import os
import re

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