{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  packages = with pkgs; [
    python312
    gcc
    gnumake

    # Python packages from nixpkgs
    python312Packages.numpy
    python312Packages.scipy
    python312Packages.pandas
    python312Packages.scikit-learn
    python312Packages.matplotlib
    python312Packages.seaborn
    python312Packages.pyyaml
    python312Packages.optuna
    python312Packages.umap-learn

    cudaPackages.cudatoolkit  
    cudaPackages.nccl 
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.glibc
    pkgs.zlib
    pkgs.libffi
    pkgs.cudaPackages.cudatoolkit
  ] + ":/run/opengl-driver/lib";

  shellHook = ''
    export PROJECT_ROOT="$(pwd)"
    export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"
    export PATH="$CUDA_HOME/bin:$PATH"
    export PATH="$PROJECT_ROOT/scripts:$PATH"

    export PIP_DISABLE_PIP_VERSION_CHECK=1 
    export PIP_PROGRESS_BAR=off

    echo "[NIX-SHELL] PROJECT_ROOT set to: $PROJECT_ROOT" 
    echo "[NIX-SHELL] initializing python environment..."

    if [ ! -d ".venv" ]; then 
      echo "[NIX-SHELL] creating new virtual environment..."
      python -m venv .venv 
    fi 

    if [ -n "$VIRTUAL_ENV" ]; then 
      deactivate || true  
    fi

    source .venv/bin/activate
    echo "[NIX-SHELL] activated virtual environment: $VIRTUAL_ENV"
    
    if [ ! -f ".venv/.bootstrapped" ]; then 
      echo "[NIX-SHELL] installing remaining packages via pip"
      PIP_QUIET="python -m pip -q"

      $PIP_QUIET install --upgrade pip || echo "[pip] upgrade failed" >&2 
      $PIP_QUIET install torch torchvision torchaudio\
        --index-url https://download.pytorch.org/whl/cu124 || 
        echo "[pip] torch install failed" >&2 
      $PIP_QUIET install torch-scatter torch-sparse \
        -f https://data.pyg.ord/whl/torch-2.6.0+cu124.html ||
        echo "[pip] pyg deps failed" >&2 

      $PIP_QUIET install torch-geometric muon-optimizer pyarrow imageio ripser persim gudhi torch-topological || 
        echo "[pip] extra deps failed" >&2 

      echo "[NIX-SHELL] installing CUDA xgboost"
      $PIP_QUIET install --no-cache-dir 'xgboost>=2.0.0' \
        --config-settings=use_cuda=ON \
        --config-settings=use_nccl=ON || echo "[pip] xgboost install failed" >&2 

      $PIP_QUIET install -e . || echo "[pip] editable install failed" >&2 
      touch .venv/.bootstrapped 
    fi 

    echo "[NIX-SHELL] injecting python development headers"
    PYTHON_INC=$(${pkgs.python312}/bin/python -c "import sysconfig; \
      print(sysconfig.get_paths()['include'])")

    export C_INCLUDE_PATH="$PYTHON_INC:''${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="$PYTHON_INC:''${CPLUS_INCLUDE_PATH:-}"
    export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"
    '';
}
