import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union


def load_json_file(file_path: str) -> Dict:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_values(json_data: Dict) -> Tuple[Union[int, float], Union[int, float]]:
    """Extract anomaly_max and anomaly_pixels values from JSON data."""
    try:
        patch_analysis = json_data.get('patch_analysis', [])
        if not patch_analysis or len(patch_analysis) == 0:
            return None, None
        
        first_item = patch_analysis[0]
        anomaly_max = first_item.get('anomaly_max')
        anomaly_pixels = first_item.get('anomaly_pixels')
        
        return anomaly_max, anomaly_pixels
    except Exception as e:
        print(f"Error extracting values: {e}")
        return None, None


def values_differ(val1: Union[int, float], val2: Union[int, float]) -> bool:
    """Compare two values, treating int and float as equal if they have the same numerical value."""
    if val1 is None or val2 is None:
        return val1 != val2
    
    # Convert to float for comparison to handle int vs float cases
    float1 = float(val1)
    float2 = float(val2)
    
    return abs(float1 - float2) > 1e-10  # Small tolerance for floating point comparison


def compare_json_files(folder1: str, folder2: str) -> List[Dict]:
    """Compare JSON files with the same names in two folders."""
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)
    
    if not folder1_path.exists():
        raise ValueError(f"Folder 1 does not exist: {folder1}")
    if not folder2_path.exists():
        raise ValueError(f"Folder 2 does not exist: {folder2}")
    
    # Get all JSON files in both folders
    files1 = {f.name: f for f in folder1_path.glob("*.json")}
    files2 = {f.name: f for f in folder2_path.glob("*.json")}
    
    # Find common files
    common_files = set(files1.keys()) & set(files2.keys())
    
    differences = []
    
    for filename in common_files:
        file1_path = files1[filename]
        file2_path = files2[filename]
        
        # Load JSON data
        data1 = load_json_file(str(file1_path))
        data2 = load_json_file(str(file2_path))
        
        if data1 is None or data2 is None:
            continue
        
        # Extract values
        max1, pixels1 = extract_values(data1)
        max2, pixels2 = extract_values(data2)
        
        # Check for differences
        max_differs = values_differ(max1, max2)
        pixels_differs = values_differ(pixels1, pixels2)
        
        if max_differs or pixels_differs:
            differences.append({
                'filename': filename,
                'folder1_path': str(file1_path),
                'folder2_path': str(file2_path),
                'anomaly_max': {
                    'folder1': max1,
                    'folder2': max2,
                    'differs': max_differs
                },
                'anomaly_pixels': {
                    'folder1': pixels1,
                    'folder2': pixels2,
                    'differs': pixels_differs
                }
            })
    
    return differences


def print_differences(differences: List[Dict], folder1_name: str, folder2_name: str):
    """Print the differences in a formatted way."""
    if not differences:
        print("No differences found between the two folders.")
        return
    
    print(f"\nFound {len(differences)} files with differences between '{folder1_name}' and '{folder2_name}':")
    print("=" * 80)
    
    for diff in differences:
        print(f"\nFile: {diff['filename']}")
        print(f"  Folder 1: {diff['folder1_path']}")
        print(f"  Folder 2: {diff['folder2_path']}")
        
        if diff['anomaly_max']['differs']:
            print(f"  anomaly_max differs:")
            print(f"    Folder 1: {diff['anomaly_max']['folder1']} ({type(diff['anomaly_max']['folder1']).__name__})")
            print(f"    Folder 2: {diff['anomaly_max']['folder2']} ({type(diff['anomaly_max']['folder2']).__name__})")
        
        if diff['anomaly_pixels']['differs']:
            print(f"  anomaly_pixels differs:")
            print(f"    Folder 1: {diff['anomaly_pixels']['folder1']} ({type(diff['anomaly_pixels']['folder1']).__name__})")
            print(f"    Folder 2: {diff['anomaly_pixels']['folder2']} ({type(diff['anomaly_pixels']['folder2']).__name__})")
        
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description='Compare JSON files with same names in two folders')
    parser.add_argument('folder1', help='Path to first folder')
    parser.add_argument('folder2', help='Path to second folder')
    parser.add_argument('--output', '-o', help='Output file to save results (optional)')
    
    args = parser.parse_args()
    
    try:
        differences = compare_json_files(args.folder1, args.folder2)
        
        # Print to console
        print_differences(differences, args.folder1, args.folder2)
        
        # Save to file if requested
        if args.output and differences:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(differences, f, indent=2, ensure_ascii=False)
                print(f"\nResults saved to: {args.output}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 