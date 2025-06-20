{
  self,
  pkgs,
  workspace,
  pythonSet,
  ...
}:

rec {

  deco-diff-env = pythonSet.mkVirtualEnv "deco-diff-env" workspace.deps.default;
  
  deco-diff = pythonSet.deco-diff;

  python = pythonSet.python;

  jupyter = pkgs.writeShellApplication {
    name = "jupyter";
    runtimeInputs = [ deco-diff-env ];
    text = ''
      exec jupyter lab "$@"
    '';
  };

  train-deco-diff = pkgs.writeShellApplication {
    name = "train-deco-diff";
    runtimeInputs = [ deco-diff-env ];
    text = ''
      train_deco_diff "$@"
    '';
  };

  torchrun-train-deco-diff = pkgs.writeShellApplication {
    name = "torchrun-train-deco-diff";
    runtimeInputs = [ deco-diff-env ];
    text = ''
      torchrun ${deco-diff-env}/bin/train_deco_diff "$@"
    '';
  };

  multi-gpu-train-example = pkgs.writeShellApplication {
    name = "multi-gpu-train-example";
    runtimeInputs = [ deco-diff-env ];
    text = ''
      export NCCL_P2P_DISABLE=1;
      time torchrun \
        --standalone \
        --nproc_per_node=2 \
        "$(which train_deco_diff)" \
          --dataset pcb \
          --data-dir ~/dataset/PCB/Huang/PCB_DATASET/PCB_gray_128 \
          --model-size UNet_L \
          --object-class all \
          --augment False \
          --ckpt-every 1 \
          --resume-dir DeCo-Diff_pcb_all_UNet_L_128_CenterCrop/001-UNet_L \
          --image-size 128 \
          --center-size 128 \
          --global-batch-size 532 \
          --epochs 10000
          # --nproc_per_node=1 \
          # --resume-dir test \
          # --global-batch-size 253 \
    '';
  };

  #wheels = pkgs.symlinkJoin {
  #  name = "wheels";
  #  paths = [
  #    # Include the main project wheel output:
  #    (pythonSet.${workspace.projectName}.override { pyprojectHook = pythonSet.pyprojectDistHook; }).dist
  #  ];# ++ (lib.attrValues (workspace.deps.default) 
  #        # ^ add all dependency wheels
  #     #   // (dep: pythonSet.${dep}.dist));
  #};
}
