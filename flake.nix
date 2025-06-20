{
  description = "DeCo Diff";

  inputs = {
    haedosa.url = "git+ssh://gitea@gitea.internal/Haedosa/flakes";

    nixpkgs.follows = "haedosa/nixpkgs";

    pyproject-nix = {
      url = "github:sepiabrown/pyproject.nix/windows-support-for-pep508";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
      };
      pkgs-win = pkgs.pkgsCross.mingwW64;

      python = pkgs.python311Full;
      python-win = pkgs-win.python311;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./project; };

      pythonSet = import ./pythonSet.nix {
        inherit
          pkgs
          python
          workspace
          pyproject-nix
          pyproject-build-systems
          ;
      };

      pythonSet-win = import ./pythonSet.nix {
        inherit
          workspace
          pyproject-nix
          pyproject-build-systems
          ;
        pkgs = pkgs-win;
        python = python-win;
        forWin = true;
      };
    in
    {
      inherit pkgs pkgs-win workspace pythonSet pythonSet-win pyproject-nix;

      packages.x86_64-linux = import ./packages {
        inherit
          self
          pkgs
          workspace
          pythonSet
          ;
      };
      packages.mingwW64= import ./packages {
        inherit
          self
          workspace
          ;
        pkgs = pkgs-win;
        pythonSet = pythonSet-win;
      };

      devShells.x86_64-linux = import ./develop.nix {
        inherit
          pkgs
          python
          workspace
          pythonSet
          ;
      };

      devShells.mingwW64 = import ./develop.nix {
        inherit
          workspace
          ;
        pkgs = pkgs-win;
        python = python-win;
        pythonSet = pythonSet-win;
      };

      apps.x86_64-linux = import ./apps {
        inherit
          self
          pkgs
          workspace
          pythonSet
          ;
      };

      formatter.x86_64-linux = pkgs.nixfmt-tree;

    };
}
