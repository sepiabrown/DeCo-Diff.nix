import json
import os
def generate_config(checkpoint, perturbation):
    base_config = {
        "dataset": "pcb",
        "data-dir": "~/datasets/PCB/Huang/PCB_DATASET/PCB-gray-128___deco-diff",
        "model-size": "UNet_L",
        "object-class": "all",
        "anomaly-class": "all",
        "image-size": "128",
        "center-size": "128",
        "center-crop": "False",
        "batch-num": "1",
        "pretrained": f"~/DeCo-Diff/DeCo-Diff_pcb_selected_UNet_L_128/001-UNet_L/checkpoints/{checkpoint}.pt",
        "perturbation": perturbation,
    }
    
    train_config = base_config.copy()
    train_config["split"] = "train"
    train_config["split-csv-path"] = "~/datasets/PCB/Huang/PCB_DATASET/PCB-gray-128___deco-diff/pcb-split___selected_train.csv"
    
    test_config = base_config.copy()
    test_config["split"] = "test"
    test_config["split-csv-path"] = "~/datasets/PCB/Huang/PCB_DATASET/PCB-gray-128___deco-diff/pcb-split___selected_test.csv"
    return {
        f"ckpt_{checkpoint}__sp_train__ptb_{perturbation}": train_config,
        f"ckpt_{checkpoint}__sp_test__ptb_{perturbation}": test_config,
        #f"ckpt_{checkpoint}__sp_train__ptb_{perturbation}_ds_pcb__dd_PCB-gray-128___deco-diff__ms_UNet_L__oc_all__ac_all__is_128__cs_128__cc_False__bn_5": train_config,
        #f"ckpt_{checkpoint}__sp_test__ptb_{perturbation}_ds_pcb__dd_PCB-gray-128___deco-diff__ms_UNet_L__oc_all__ac_all__is_128__cs_128__cc_False__bn_5": test_config
    }

def generate_config2(checkpoint, perturbation, split):
    base_config = {
        "dataset": "pcb",
        "data-dir": "~/datasets/PCB/Huang/PCB_DATASET/PCB-gray-128___deco-diff",
        "model-size": "UNet_L",
        "object-class": "all",
        "anomaly-class": "all",
        "image-size": "128",
        "center-size": "128",
        "center-crop": "False",
        "batch-num": "1",
        "pretrained": f"~/DeCo-Diff/DeCo-Diff_pcb_selected_UNet_L_128/001-UNet_L/checkpoints/{checkpoint}.pt",
        "perturbation": perturbation,
        "split": split,
        "split-csv-path": f"~/datasets/PCB/Huang/PCB_DATASET/PCB-gray-128___deco-diff/pcb-split___selected_{split}.csv"
    }
    
    return {
        f"ckpt_{checkpoint}__sp_{split}__ptb_{perturbation}": base_config,
    }

for perturbation in ["brightness", "shift_x"]:
    configs_train = {}
    configs_test = {}
    for checkpoint in range(26000, 0, -5000):
        configs_train.update(generate_config2(checkpoint, perturbation, "train"))
        configs_test.update(generate_config2(checkpoint, perturbation, "test"))
    for checkpoint in range(26000, 0, -1000):
        if (checkpoint - 1000) % 5000 != 0:
            configs_train.update(generate_config2(checkpoint, perturbation, "train"))
            configs_test.update(generate_config2(checkpoint, perturbation, "test"))
    for checkpoint in range(26000, 0, -500):
        if checkpoint % 1000 != 0:
            configs_train.update(generate_config2(checkpoint, perturbation, "train"))
            configs_test.update(generate_config2(checkpoint, perturbation, "test"))
    path_train = f"input_json/250624_{perturbation}_train.json"
    path_test = f"input_json/250624_{perturbation}_test.json"
    os.makedirs(os.path.dirname(path_train), exist_ok=True)
    os.makedirs(os.path.dirname(path_test), exist_ok=True)
    # Write to JSON file
    with open(path_train, 'w') as f:
        json.dump(configs_train, f, indent=2) 
    with open(path_test, 'w') as f:
        json.dump(configs_test, f, indent=2) 