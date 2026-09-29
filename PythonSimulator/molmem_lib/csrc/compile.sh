#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check for CUDA installation
if [ -z "$CUDA_HOME" ]; then
    for candidate in /usr/local/cuda /usr/local/cuda-* /opt/cuda; do
        if [ -d "$candidate" ] && [ -f "$candidate/bin/nvcc" ]; then
            export CUDA_HOME="$candidate"
            export CUDA_PATH="$candidate"
            export PATH="$candidate/bin:$PATH"
            break
        fi
    done
fi

# 2. Check for ROCm / HIP installation
if [ -z "$HIP_PATH" ] && [ -z "$ROCM_PATH" ]; then
    for candidate in /opt/rocm* /usr/local/rocm*; do
        if [ -d "$candidate" ] && ([ -f "$candidate/bin/hipcc" ] || [ -d "$candidate/include" ]); then
            export ROCM_PATH="$candidate"
            export HIP_PATH="$candidate"
            export ROCM_HOME="$candidate"
            export PATH="$candidate/bin:$PATH"
            break
        fi
    done
fi

# 3. Select active Python executable (molmem-env virtualenv or system python)
if [ -f "$SCRIPT_DIR/../../molmem-env/bin/python" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/../../molmem-env/bin/python"
elif [ -f "$SCRIPT_DIR/../../molmem-env/Scripts/python.exe" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/../../molmem-env/Scripts/python.exe"
else
    PYTHON_EXEC="$(command -v python3 || command -v python)"
fi

echo "=========================================================================="
echo "MOLMEM: Compiling native GPU solver extension via $PYTHON_EXEC"
echo "=========================================================================="

"$PYTHON_EXEC" setup.py build_ext --inplace
echo "[+] Build complete!"
