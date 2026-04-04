{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  # 1. Fetch Eelco's pre-compiled Blender (CUDA/OptiX ready)
  blenderFlake = builtins.getFlake "github:edolstra/nix-warez?dir=blender";
  blenderPkg = blenderFlake.packages.${builtins.currentSystem}.default;

  # 2. Setup your Python environment for the dataset generation script
  # FIX: Changed python311 to python3 to resolve Sphinx version incompatibilities
  myPython = pkgs.python3.withPackages (ps: with ps; [
    datasets
    pillow
    requests
  ]);

  # 3. Create an FHS (Filesystem Hierarchy Standard) environment
  fhs = pkgs.buildFHSEnv {
    name = "blender-gpu-env";

    # Packages to install INSIDE the fake environment
    targetPkgs = pkgs: (with pkgs; [
      blenderPkg
      myPython

      # C-libraries the binary was crashing over
      glib
      gvfs
      vulkan-loader
      wayland
      libxkbcommon
      libx11  # FIX: Updated from xorg.libX11 to silence the warning
    ]);

    # Bind the NVIDIA drivers and force the binary to see them
    profile = ''
      # Expose NixOS system NVIDIA/OpenGL drivers
      export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH"
      export LIBGL_DRIVERS_PATH="/run/opengl-driver/lib/dri"

      # Force NVIDIA Offloading (just in case you are on a hybrid laptop setup)
      export __NV_PRIME_RENDER_OFFLOAD=1
      export __GLX_VENDOR_LIBRARY_NAME=nvidia

      echo "=========================================================="
      echo " Blender Pre-compiled Binary + RTX 4060 Environment Loaded!"
      echo "=========================================================="
    '';

    runScript = "bash";
  };

in
fhs.env