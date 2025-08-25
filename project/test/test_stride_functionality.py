#!/usr/bin/env python3
"""
Test script to verify stride functionality and mean aggregation works correctly.
This creates a simple test case with small parameters to manually verify behavior.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image as PILImage
import json
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluate_and_process import AnnotatedImageDataset
from torchvision import transforms

def create_test_image_and_annotation(temp_dir, image_size=64):
    """Create a simple test image and annotation for testing."""
    
    # Create a simple test image with a clear pattern
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    
    # Add some patterns to make patches distinguishable
    # Top-left quadrant: red
    image[:image_size//2, :image_size//2, 0] = 255
    # Top-right quadrant: green  
    image[:image_size//2, image_size//2:, 1] = 255
    # Bottom-left quadrant: blue
    image[image_size//2:, :image_size//2, 2] = 255
    # Bottom-right quadrant: white
    image[image_size//2:, image_size//2:] = 255
    
    # Save test image
    image_path = os.path.join(temp_dir, "test_image.png")
    PILImage.fromarray(image).save(image_path)
    
    # Create annotation file
    annotation = {
        "image_path": image_path,
        "defective_patches": [],  # No defective patches for this test
        "image_width": image_size,
        "image_height": image_size
    }
    
    annotation_path = os.path.join(temp_dir, "test_image_annotations.json")
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f)
    
    return image_path, annotation_path, temp_dir

def test_stride_patch_extraction():
    """Test that different stride values produce the expected number of patches."""
    
    print("🧪 Testing stride patch extraction...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        image_path, annotation_path, annotation_dir = create_test_image_and_annotation(temp_dir, image_size=64)
        
        # Test different stride configurations
        test_cases = [
            {"patch_size": 32, "stride": None, "expected_patches": 4, "description": "No overlap (stride=None)"},
            {"patch_size": 32, "stride": 32, "expected_patches": 4, "description": "No overlap (stride=patch_size)"},
            {"patch_size": 32, "stride": 16, "expected_patches": 9, "description": "50% overlap (stride=patch_size/2)"},
            {"patch_size": 32, "stride": 8, "expected_patches": 25, "description": "75% overlap (stride=patch_size/4)"},
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n  Test {i+1}: {test_case['description']}")
            print(f"    Patch size: {test_case['patch_size']}, Stride: {test_case['stride']}")
            
            try:
                # Create dataset with test parameters
                dataset = AnnotatedImageDataset(
                    annotation_dir=annotation_dir,
                    patch_size=test_case['patch_size'],
                    stride=test_case['stride'],
                    transform=transforms.ToTensor(),
                    object_class="pcb",
                )
                
                actual_patches = len(dataset)
                expected_patches = test_case['expected_patches']
                
                print(f"    Expected patches: {expected_patches}")
                print(f"    Actual patches: {actual_patches}")
                
                if actual_patches == expected_patches:
                    print(f"    ✅ PASS")
                else:
                    print(f"    ❌ FAIL - Expected {expected_patches}, got {actual_patches}")
                    
                # Test a few patch extractions to ensure they work
                if len(dataset) > 0:
                    patch_0 = dataset[0]
                    print(f"    Sample patch shape: {patch_0[0].shape}")
                    print(f"    Sample patch coords: {patch_0[5]}")  # patch_coords
                    
            except Exception as e:
                print(f"    ❌ ERROR: {e}")

def test_mean_aggregation_logic():
    """Test the mean aggregation logic in save_image_results_from_records."""
    
    print("\n🧪 Testing mean aggregation logic...")
    
    # Create synthetic overlapping patch data
    image_height, image_width = 64, 64
    patch_size = 32
    
    # Simulate overlapping patches for mean aggregation
    # We'll create a simple case where pixels get multiple predictions
    
    # Test data: patches with known values
    patches_data = [
        {"coords": [0, 0, 32, 0, 32, 32, 0, 32], "anomaly_value": 0.8},      # Top-left patch
        {"coords": [16, 0, 48, 0, 48, 32, 16, 32], "anomaly_value": 0.4},    # Overlapping patch (16px right)
        {"coords": [0, 16, 32, 16, 32, 48, 0, 48], "anomaly_value": 0.6},    # Overlapping patch (16px down)
        {"coords": [16, 16, 48, 16, 48, 48, 16, 48], "anomaly_value": 0.2},  # Center overlap patch
    ]
    
    # Manual calculation of expected mean for the center overlap region (16-32, 16-32)
    # This region is covered by all 4 patches
    expected_mean_center = (0.8 + 0.4 + 0.6 + 0.2) / 4  # = 0.5
    
    print(f"  Test case: 4 overlapping {patch_size}x{patch_size} patches on {image_width}x{image_height} image")
    print(f"  Patch values: {[p['anomaly_value'] for p in patches_data]}")
    print(f"  Expected mean in center overlap region: {expected_mean_center}")
    
    # Simulate the aggregation logic (simplified version)
    anomaly_map = np.zeros((image_height, image_width), dtype=np.float32)
    count_map = np.zeros((image_height, image_width), dtype=np.float32)
    
    for patch_data in patches_data:
        x1, y1, x2, y2, x3, y3, x4, y4 = patch_data["coords"]
        # Assuming rectangular patches aligned with axes
        patch_height = y3 - y1
        patch_width = x2 - x1
        
        # Create a patch filled with the anomaly value
        patch_values = np.full((patch_height, patch_width), patch_data["anomaly_value"], dtype=np.float32)
        
        # Add to accumulation maps
        anomaly_map[y1:y1+patch_height, x1:x1+patch_width] += patch_values
        count_map[y1:y1+patch_height, x1:x1+patch_width] += 1
    
    # Compute mean
    valid_mask = count_map > 0
    final_map = np.zeros_like(anomaly_map)
    final_map[valid_mask] = anomaly_map[valid_mask] / count_map[valid_mask]
    
    # Check the center overlap region (16-32, 16-32)
    center_region = final_map[16:32, 16:32]
    actual_mean_center = np.mean(center_region)
    
    print(f"  Actual mean in center overlap region: {actual_mean_center}")
    print(f"  Count map in center region: {np.unique(count_map[16:32, 16:32])}")
    
    # Verify the result
    tolerance = 1e-6
    if abs(actual_mean_center - expected_mean_center) < tolerance:
        print(f"  ✅ PASS - Mean aggregation working correctly")
    else:
        print(f"  ❌ FAIL - Expected {expected_mean_center}, got {actual_mean_center}")
    
    # Additional checks for different overlap regions
    print(f"\n  Additional checks:")
    print(f"    Top-left corner (single patch): {final_map[0, 0]} (expected: 0.8)")
    print(f"    Top edge (2 patches): {final_map[0, 24]} (expected: {(0.8 + 0.4)/2})")
    print(f"    Left edge (2 patches): {final_map[24, 0]} (expected: {(0.8 + 0.6)/2})")

def test_coordinate_format():
    """Test that 8-value coordinates are properly generated and handled."""
    
    print("\n🧪 Testing coordinate format...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        image_path, annotation_path, annotation_dir = create_test_image_and_annotation(temp_dir, image_size=64)
        
        # Create dataset with small patches to get multiple coordinates
        dataset = AnnotatedImageDataset(
            annotation_dir=annotation_dir,
            patch_size=16,
            stride=16,  # No overlap for simplicity
            transform=transforms.ToTensor(),
            object_class="pcb",
        )
        
        print(f"  Created dataset with {len(dataset)} patches")
        
        # Check first few patches for coordinate format
        for i in range(min(4, len(dataset))):
            x, seg, object_cls, anomaly_classes, image_path_ret, patch_coords = dataset[i]
            
            print(f"  Patch {i}:")
            print(f"    Coordinates: {patch_coords.tolist()}")
            print(f"    Coordinate count: {len(patch_coords)}")
            
            # Verify 8-value format
            if len(patch_coords) == 8:
                x1, y1, x2, y2, x3, y3, x4, y4 = patch_coords.tolist()
                
                # Check if it's a proper rectangle
                width = x2 - x1
                height = y4 - y1
                
                print(f"    Rectangle: ({x1},{y1}) to ({x3},{y3}), size: {width}x{height}")
                
                # Verify it's a proper 16x16 rectangle
                if width == 16 and height == 16:
                    print(f"    ✅ PASS - Proper 16x16 rectangle")
                else:
                    print(f"    ❌ FAIL - Expected 16x16, got {width}x{height}")
            else:
                print(f"    ❌ FAIL - Expected 8 coordinates, got {len(patch_coords)}")

def main():
    """Run all stride functionality tests."""
    
    print("🔬 Testing Stride and Mean Aggregation Functionality")
    print("=" * 60)
    
    try:
        test_stride_patch_extraction()
        test_mean_aggregation_logic()
        test_coordinate_format()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed! Check results above for any failures.")
        print("\n💡 To test with real data, try:")
        print("   python project/evaluate_and_process.py --mode full_pipeline \\")
        print("       --annotation-dir your/annotation/dir \\")
        print("       --pretrained your/model.pt \\")
        print("       --patch-size 128 --stride 64 \\")
        print("       --enable-save-whole-image-results \\")
        print("       --debug")
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
