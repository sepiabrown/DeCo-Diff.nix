{
  self,
  pkgs,
  workspace,
  pythonSet,
  ...
}:

rec {

  deco-diff-env = pythonSet.mkVirtualEnv "deco-diff-env" workspace.deps.default;

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
}
