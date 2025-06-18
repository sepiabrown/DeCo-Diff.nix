import json
import os
def generate_config(checkpoint, perturbation):
    base_config = {
        "dataset": "pcb",
        "data-dir": "~/dataset/PCB/Huang/PCB_DATASET/PCB-gray-128___deco-diff",
        "model-size": "UNet_L",
        "object-class": "all",
        "anomaly-class": "all",
        "image-size": "128",
        "center-size": "128",
        "center-crop": "False",
        "batch-num": "5",
        "pretrained": f"DeCo-Diff_pcb_all_UNet_L_128_CenterCrop/001-UNet_L/checkpoints/{checkpoint}.pt",
        "perturbation": perturbation
    }
    
    train_config = base_config.copy()
    train_config["split"] = "train"
    
    test_config = base_config.copy()
    test_config["split"] = "test"
    
    return {
        f"ptd_{checkpoint}__sp_train__ptb_{perturbation}": train_config,
        f"ptd_{checkpoint}__sp_test__ptb_{perturbation}": test_config,
        #f"ptd_{checkpoint}__sp_train__ptb_{perturbation}_ds_pcb__dd_PCB-gray-128___deco-diff__ms_UNet_L__oc_all__ac_all__is_128__cs_128__cc_False__bn_5": train_config,
        #f"ptd_{checkpoint}__sp_test__ptb_{perturbation}_ds_pcb__dd_PCB-gray-128___deco-diff__ms_UNet_L__oc_all__ac_all__is_128__cs_128__cc_False__bn_5": test_config
    }

for perturbation in ["noise"]:
    all_configs = {}
    all_configs.update(generate_config(39000, perturbation))
    all_configs.update(generate_config(38500, perturbation))
    for checkpoint in range(29000, 0, -1000):
        all_configs.update(generate_config(checkpoint, perturbation))
    path = f"input_json/{perturbation}_29000_1000.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write to JSON file
    with open(path, 'w') as f:
        json.dump(all_configs, f, indent=2) 