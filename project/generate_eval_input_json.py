import json
import os
import argparse
from itertools import product
from typing import Dict, List, Any, Union

def generate_base_config(checkpoint: str, perturbation: str | None = None) -> Dict[str, Any]:
    """Generate base configuration with common parameters."""
    base_config = {
        "dataset": "pcb",
        "annotation-dir": "annotations",
        "model-size": "UNet_L",
        "object-class": "all",
        "anomaly-class": "all",
        "image-size": "128",
        "center-size": "128",
        "center-crop": "False",
        "batch-num": "100000",
        "pretrained": f"C:/Users/Public/Documents/deco-diff/DeCo-Diff_pcb_selected_UNet_L_128_250624/001-UNet_L/checkpoints/{checkpoint}.pt",
    }
    
    if perturbation:
        base_config["perturbation"] = perturbation
    
    return base_config

def generate_config_with_loops(
    checkpoint: str,
    split: str,
    perturbation: str | None = None,
    loop_args: Dict[str, List[Any]] | None = None,
    fixed_args: Dict[str, Any] | None = None
) -> Dict[str, Dict[str, Any]]:
    """
    Generate configurations by looping through specified arguments.
    
    Args:
        checkpoint: Model checkpoint name
        split: Data split ('train' or 'test')
        perturbation: Optional perturbation type
        loop_args: Dictionary of argument names to lists of values to loop through
        fixed_args: Dictionary of fixed argument values
    
    Returns:
        Dictionary of configuration names to configurations
    """
    base_config = generate_base_config(checkpoint, perturbation)
    
    # Add split-specific arguments
    base_config["split"] = split
    base_config["split-csv-path"] = f"~/datasets/PCB/Huang/PCB_DATASET/PCB-gray-128___deco-diff/pcb-split___selected_{split}.csv"
    
    # Add fixed arguments
    if fixed_args:
        base_config.update(fixed_args)
    
    configs = {}
    
    if not loop_args:
        # No looping, just return single config
        config_name_parts = []
        if perturbation:
            config_name_parts.append(f"perturbation_{perturbation}")
        config_name = "__".join(config_name_parts) if config_name_parts else "config"
        configs[config_name] = base_config
        return configs
    
    # Generate all combinations of loop arguments
    arg_names = list(loop_args.keys())
    arg_values = list(loop_args.values())
    
    for combination in product(*arg_values):
        # Create config for this combination
        config = base_config.copy()
        
        # Build config name with only looped values
        config_name_parts = []
        
        # Only add checkpoint if it's being looped
        if 'checkpoint' in loop_args:
            checkpoint_idx = arg_names.index('checkpoint')
            config_name_parts.append(f"checkpoint_{combination[checkpoint_idx]}")
        
        # Only add split if it's being looped
        if 'split' in loop_args:
            split_idx = arg_names.index('split')
            config_name_parts.append(f"split_{combination[split_idx]}")
        
        # Add perturbation if specified
        if perturbation:
            config_name_parts.append(f"perturbation_{perturbation}")
        
        # Add loop argument values to config and name
        for arg_name, arg_value in zip(arg_names, combination):
            config[arg_name] = arg_value
            # Skip checkpoint and split as they're handled above
            if arg_name in ['checkpoint', 'split']:
                continue
            # Make the looped values more prominent in the name
            config_name_parts.append(f"{arg_name}_{arg_value}")
        
        # Join all parts to create the config name
        config_name = "__".join(config_name_parts) if config_name_parts else "config"
        
        configs[config_name] = config
    
    return configs

def parse_range_arg(arg_str: str) -> List[Union[int, float, str]]:
    """
    Parse range argument in format: 'start:end:step' or 'value1,value2,value3'
    
    Examples:
        '5:255:5' -> [5, 10, 15, ..., 250, 255]
        '0:128:10' -> [0, 10, 20, ..., 120, 128]
        'annotations,annotations_test' -> ['annotations', 'annotations_test']
    """
    if ':' in arg_str:
        # Range format: start:end:step
        parts = arg_str.split(':')
        if len(parts) == 3:
            start, end, step = map(float, parts)
            return list(range(int(start), int(end) + int(step), int(step)))
        else:
            raise ValueError(f"Invalid range format: {arg_str}. Use 'start:end:step'")
    else:
        # Comma-separated values
        return [val.strip() for val in arg_str.split(',')]

def main():
    parser = argparse.ArgumentParser(description='Generate evaluation input JSON with flexible argument looping')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint name')
    parser.add_argument('--split', type=str, choices=['train', 'test'], required=True, help='Data split')
    parser.add_argument('--perturbation', type=str, help='Perturbation type (optional)')
    parser.add_argument('--output-dir', type=str, default='input_json', help='Output directory for JSON files')
    
    # Add argument for each parameter that can be looped
    parser.add_argument('--anomaly-pixel-num-threshold', type=str, 
                       help='Range or values for anomaly-pixel-num-threshold (e.g., "5:255:5" or "10,20,30")')
    parser.add_argument('--anomaly-binary-threshold', type=str,
                       help='Range or values for anomaly-binary-threshold (e.g., "0:128:10" or "5,10,15")')
    parser.add_argument('--annotation-dir', type=str,
                       help='Values for annotation-dir (e.g., "annotations,annotations_test")')
    
    # Add more arguments as needed
    parser.add_argument('--reverse-steps', type=str,
                       help='Range or values for reverse-steps (e.g., "1:10:1" or "3,5,7")')
    parser.add_argument('--batch-num', type=str,
                       help='Range or values for batch-num (e.g., "1:5:1" or "1,2,3")')
    
    # Fixed arguments (not looped)
    parser.add_argument('--dataset', type=str, help='Fixed dataset value')
    parser.add_argument('--data-dir', type=str, help='Fixed data directory')
    parser.add_argument('--model-size', type=str, help='Fixed model size')
    parser.add_argument('--object-class', type=str, help='Fixed object class')
    parser.add_argument('--anomaly-class', type=str, help='Fixed anomaly class')
    parser.add_argument('--image-size', type=str, help='Fixed image size')
    parser.add_argument('--center-size', type=str, help='Fixed center size')
    parser.add_argument('--center-crop', type=str, help='Fixed center crop value')
    parser.add_argument('--pretrained', type=str, help='Fixed pretrained path')
    
    args = parser.parse_args()
    
    # Build loop arguments dictionary
    loop_args = {}
    if args.anomaly_pixel_num_threshold:
        loop_args['anomaly-pixel-num-threshold'] = parse_range_arg(args.anomaly_pixel_num_threshold)
    if args.anomaly_binary_threshold:
        loop_args['anomaly-binary-threshold'] = parse_range_arg(args.anomaly_binary_threshold)
    if args.annotation_dir:
        loop_args['annotation-dir'] = parse_range_arg(args.annotation_dir)
    if args.reverse_steps:
        loop_args['reverse-steps'] = parse_range_arg(args.reverse_steps)
    if args.batch_num:
        loop_args['batch-num'] = parse_range_arg(args.batch_num)
    
    # Build fixed arguments dictionary
    fixed_args = {}
    if args.dataset:
        fixed_args['dataset'] = args.dataset
    if args.data_dir:
        fixed_args['data-dir'] = args.data_dir
    if args.model_size:
        fixed_args['model-size'] = args.model_size
    if args.object_class:
        fixed_args['object-class'] = args.object_class
    if args.anomaly_class:
        fixed_args['anomaly-class'] = args.anomaly_class
    if args.image_size:
        fixed_args['image-size'] = args.image_size
    if args.center_size:
        fixed_args['center-size'] = args.center_size
    if args.center_crop:
        fixed_args['center-crop'] = args.center_crop
    if args.pretrained:
        fixed_args['pretrained'] = args.pretrained
    
    # Generate configurations
    configs = generate_config_with_loops(
        checkpoint=args.checkpoint,
        split=args.split,
        perturbation=args.perturbation,
        loop_args=loop_args,
        fixed_args=fixed_args
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate filename with only looped values
    filename_parts = []
    
    # Only add checkpoint if it's being looped
    if 'checkpoint' in loop_args:
        checkpoint_values = loop_args['checkpoint']
        if len(checkpoint_values) <= 3:
            values_str = "_".join(str(v) for v in checkpoint_values)
            filename_parts.append(f"checkpoint_{values_str}")
        else:
            values_str = f"{checkpoint_values[0]}-{checkpoint_values[-1]}"
            filename_parts.append(f"checkpoint_{values_str}")
    
    # Only add split if it's being looped
    if 'split' in loop_args:
        split_values = loop_args['split']
        if len(split_values) <= 3:
            values_str = "_".join(str(v) for v in split_values)
            filename_parts.append(f"split_{values_str}")
        else:
            values_str = f"{split_values[0]}-{split_values[-1]}"
            filename_parts.append(f"split_{values_str}")
    
    # Add perturbation if specified
    if args.perturbation:
        filename_parts.append(f"perturbation_{args.perturbation}")
    
    # Add looped values to filename
    if loop_args:
        for arg_name, values in loop_args.items():
            # Skip checkpoint and split as they're handled above
            if arg_name in ['checkpoint', 'split']:
                continue
                
            if len(values) <= 3:  # If few values, list them all
                values_str = "_".join(str(v) for v in values)
                filename_parts.append(f"{arg_name}_{values_str}")
            else:  # If many values, show range
                if isinstance(values[0], (int, float)) and len(values) > 1:
                    step = values[1] - values[0] if len(values) > 1 else 1
                    values_str = f"{values[0]}-{values[-1]}-{step}"
                else:
                    values_str = f"{len(values)}_values"
                filename_parts.append(f"{arg_name}_{values_str}")
    
    # If no looped values, create a simple filename
    if not filename_parts:
        filename_parts = ["config"]
    
    filename = f"{'_'.join(filename_parts)}.json"
    output_path = os.path.join(args.output_dir, filename)
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(configs, f, indent=2)
    
    print(f"Generated {len(configs)} configurations")
    print(f"Saved to: {output_path}")
    
    # Print summary
    if loop_args:
        print("\nLoop arguments:")
        for arg_name, values in loop_args.items():
            if len(values) <= 10:  # Show all values if 10 or fewer
                print(f"  {arg_name}: {values}")
            else:  # Show range for many values
                print(f"  {arg_name}: {len(values)} values from {values[0]} to {values[-1]} (step {values[1] - values[0] if len(values) > 1 else 1})")
    
    if fixed_args:
        print("\nFixed arguments:")
        for arg_name, value in fixed_args.items():
            print(f"  {arg_name}: {value}")

# Example usage functions for backward compatibility
def generate_config(checkpoint, perturbation):
    """Backward compatibility function."""
    base_config = generate_base_config(checkpoint, perturbation)
    
    train_config = base_config.copy()
    train_config["split"] = "train"
    train_config["split-csv-path"] = "~/datasets/PCB/Huang/PCB_DATASET/PCB-gray-128___deco-diff/pcb-split___selected_train.csv"
    
    test_config = base_config.copy()
    test_config["split"] = "test"
    test_config["split-csv-path"] = "~/datasets/PCB/Huang/PCB_DATASET/PCB-gray-128___deco-diff/pcb-split___selected_test.csv"
    
    return {
        f"ckpt_{checkpoint}__sp_train__ptb_{perturbation}": train_config,
        f"ckpt_{checkpoint}__sp_test__ptb_{perturbation}": test_config,
    }

def generate_config2(checkpoint, perturbation, split):
    """Backward compatibility function."""
    return generate_config_with_loops(checkpoint, split, perturbation)

if __name__ == "__main__":
    main() 