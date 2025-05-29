{
  self,
  pkgs,
  workspace,
  pythonSet,
  ...
}:

{

  default = pythonSet.mkVirtualEnv "ddpm-ood-env" workspace.deps.default;

  jupyter = {
    type = "app";
    program = "${
      pkgs.writeShellApplication {
        name = "start-jupyter";
        runtimeInputs = [ self.apps.x86_64-linux.default ];
        text = ''
          exec jupyter lab "$@"
        '';
      }
    }/bin/start-jupyter";
  };

  multi-gpu-train-example = {
    type = "app";
    program = "${
      pkgs.writeShellApplication {
        name = "multi-gpu-train-example";
        runtimeInputs = [ self.apps.x86_64-linux.default ];
        text = ''
          export NCCL_P2P_DISABLE=1;
          time torchrun \
            --standalone \
            --nproc_per_node=2 \
            "$(which train_deco_diff)" \
            --dataset pcb \
            --data-dir ~/dataset/PCB/Huang/PCB_DATASET/PCB_gray_128 \
            --model-size UNet_L \
            --object-category all \
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
      }
    }/bin/multi-gpu-train-example";
  };
}
