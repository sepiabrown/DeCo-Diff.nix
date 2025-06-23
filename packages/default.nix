{
  self,
  pkgs,
  workspace,
  pythonSet,
  ...
}:
let
  deco-diff-env = pythonSet.mkVirtualEnv "deco-diff-env" workspace.deps.default;
  dep_srcs = __filter (src: __typeOf src != "path") (map (d: d.src) deco-diff-env.buildInputs);
in
rec {
  inherit deco-diff-env;
  
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

  wheels = pkgs.linkFarm "wheels" (map (src: {
    name = src.name;
    path = src;
  }) dep_srcs);
}
