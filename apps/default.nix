{
  self,
  pkgs,
  workspace,
  pythonSet,
  ...
}:

{

  default = pythonSet.mkVirtualEnv "ddpm-ood-env" workspace.deps.default;

}
