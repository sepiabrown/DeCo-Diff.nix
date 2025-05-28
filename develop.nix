{
  pkgs,
  python,
  workspace,
  pythonSet,
  ...
}:
let
  inherit (pkgs) lib;
in
{

  # It is of course perfectly OK to keep using an impure virtualenv workflow and only use uv2nix to build packages.
  # This devShell simply adds Python and undoes the dependency leakage done by Nixpkgs Python infrastructure.
  impure = pkgs.mkShell {
    packages = [
      python
      pkgs.uv
    ];
    env =
      {
        # Prevent uv from managing Python downloads
        UV_PYTHON_DOWNLOADS = "never";
        # Force uv to use nixpkgs Python interpreter
        UV_PYTHON = python.interpreter;
      }
      // lib.optionalAttrs pkgs.stdenv.isLinux {
        # Python libraries often load native shared objects using dlopen(3).
        # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
        LD_LIBRARY_PATH = lib.makeLibraryPath (
          __filter (p: p.pname != "glibc") pkgs.pythonManylinuxPackages.manylinux1
        );
      };
    shellHook = ''
      unset PYTHONPATH
    '';
  };

  uv2nix =
    let

      editableOverlay = workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; };

      editablePythonSet = pythonSet.overrideScope (
        lib.composeManyExtensions [
          editableOverlay

          (final: prev: {
            deco-diff = prev.deco-diff.overrideAttrs (old: {
              src = lib.sources.sourceFilesBySuffices old.src [
                ".py"
                "README.md"
                "pyproject.toml"
                "uv.lock"
              ];
              nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem { editables = [ ]; };
            });

          })
        ]
      );
      virtualenv = editablePythonSet.mkVirtualEnv "deco-diff-dev-env" workspace.deps.all;

    in
    pkgs.mkShell {
      packages = [
        virtualenv
        pkgs.uv
      ];

      env = {
        UV_NO_SYNC = "1";
        UV_PYTHON = "${virtualenv}/bin/python";
        UV_PYTHON_DOWNLOADS = "never";
      };

      shellHook = ''
        unset PYTHONPATH
        export REPO_ROOT=$(git rev-parse --show-toplevel)/project
      '';
    };
}
