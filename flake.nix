{
  description = "DeCo Diff";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
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

      python = pkgs.python311Full;

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

    in
    {
      inherit pkgs;

      packages.x86_64-linux = import ./packages {
        inherit
          self
          pkgs
          workspace
          pythonSet
          ;
      };

      devShells.x86_64-linux = import ./develop.nix {
        inherit
          pkgs
          python
          workspace
          pythonSet
          ;
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
