{
  pkgs,
  python,
  workspace,
  pyproject-nix,
  pyproject-build-systems,
  ...
}:

let

  inherit (pkgs) lib;

  # Create package overlay from workspace.
  overlay = workspace.mkPyprojectOverlay {
    # Prefer prebuilt binary wheels as a package source.
    # Sdists are less likely to "just work" because of the metadata missing from uv.lock.
    # Binary wheels are more likely to, but may still require overrides for library dependencies.
    sourcePreference = "wheel"; # or sourcePreference = "sdist";
    # Optionally customise PEP 508 environment
    # environ = {
    #   platform_release = "5.10.65";
    # };
  };

  cudaLibs = map lib.getDev [
    pkgs.cudaPackages.cudnn
    pkgs.cudaPackages.nccl
    pkgs.cudaPackages.cudatoolkit
    pkgs.cudaPackages.cuda_cupti
    pkgs.cudaPackages.cudnn_8_9
    pkgs.cudaPackages.cutensor
    pkgs.cudaPackages.cusparselt
    pkgs.cudaPackages.libcublas
    pkgs.cudaPackages.libcusparse
    pkgs.cudaPackages.libcusolver
    pkgs.cudaPackages.libcurand
    pkgs.cudaPackages.cuda_gdb
    pkgs.cudaPackages.cuda_nvcc
    pkgs.cudaPackages.cuda_cudart
    pkgs.cudaPackages.libnvjitlink
    pkgs.cudaPackages.cudatoolkit
  ];

  pyprojectOverrides = final: prev: {
    numba = prev.numba.overrideAttrs (old: {
      nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.tbb_2022_0 ];
    });
    torchaudio = prev.torchaudio.overrideAttrs (old: {
      nativeBuildInputs =
        (old.nativeBuildInputs or [ ])
        ++ cudaLibs
        ++ [
          pkgs.ffmpeg_6.dev
          pkgs.sox
        ];
      preFixup = ''
        rm $out/${final.python.sitePackages}/torio/lib/{lib,_}torio_ffmpeg{4,5}.*
        addAutoPatchelfSearchPath "${final.torch}"
      '';
    });
    torchvision = prev.torchvision.overrideAttrs (old: {
      nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ cudaLibs;
      preFixup = ''
        addAutoPatchelfSearchPath "${final.torch}"
      '';
    });
    nvidia-cudnn-cu11 = prev.nvidia-cudnn-cu11.overrideAttrs (old: {
      nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ cudaLibs;
    });
    nvidia-cusolver-cu12 = prev.nvidia-cusolver-cu12.overrideAttrs (old: {
      nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ cudaLibs;
    });
    nvidia-cusparse-cu12 = prev.nvidia-cusparse-cu12.overrideAttrs (old: {
      nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ cudaLibs;
    });
    torch = prev.torch.overrideAttrs (old: {
      nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ cudaLibs;
    });
  };

  buildSystemOverrides =
    final: prev:
    let
      deps = {
        fire = {
          setuptools = [ ];
        };
        antlr4-python3-runtime = {
          setuptools = [ ];
        };
      };
    in
    pkgs.lib.mapAttrs (
      name: spec:
      prev.${name}.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem spec;
      })
    ) deps;

  # Construct package set
  pythonSet =
    # Use base package set from pyproject.nix builders
    (pkgs.callPackage pyproject-nix.build.packages {
      inherit python;
    }).overrideScope
      (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          overlay
          pyprojectOverrides
          buildSystemOverrides
        ]
      );

in
pythonSet
