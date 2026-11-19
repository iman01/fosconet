# Installation

FOSCo-Net needs PyTorch and spconv. Labelling raw scans additionally needs MinkowskiEngine, which runs the
pretrained DigiForests segmentation network. The steps below were tested on Ubuntu 22.04 with Python 3.10,
PyTorch 2.6.0 (CUDA 12.4), spconv 2.3.8 and MinkowskiEngine 0.5.4 on an NVIDIA RTX 4090 Laptop GPU.

## 1. Environment, PyTorch and spconv

```bash
conda create -n fosconet python=3.10 -y
conda activate fosconet
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install spconv-cu124                 # match your CUDA version
pip install -e .                         # from the repository root
```

This is enough for `fosconet predict` on prepared samples and for the Python API on labelled scans.

## 2. MinkowskiEngine (for `fosconet prepare` and `fosconet run`)

MinkowskiEngine 0.5.4 has to be built from source for PyTorch 2. The CUDA development files must match the CUDA
version of PyTorch. With conda, pin `cuda-version`, otherwise newer sub-packages are pulled in:

```bash
conda install -y -c nvidia "cuda-version=12.4" cuda-nvcc cuda-cudart-dev cuda-cccl libcublas-dev libcusparse-dev
conda install -y -c conda-forge openblas
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
```

Recent CUDA 12 toolkits (CCCL) need a few missing thrust includes and one change in `reserve()` of
`src/coordinate_map_gpu.cuh`, where assigning a `std::unique_ptr` to a `std::shared_ptr` is ambiguous. Apply the
patch from this repository:

```bash
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
git apply /path/to/fosconet/docs/minkowskiengine_cuda12.patch
```

Then build:

```bash
export CUDA_HOME=$CONDA_PREFIX
python setup.py install --blas=openblas --force_cuda
python -c "import MinkowskiEngine as ME; print(ME.__version__)"
```

## 3. Segmentation checkpoint

```bash
fosconet download segmenter
```

downloads the pretrained `forest_pan_seg` checkpoint of the DigiForests authors (Malladi et al., 2025) to
`~/.cache/fosconet/` and checks its SHA-256. `--segmenter PATH` points the commands to another copy.

## 4. Check

```bash
pip install pytest
pytest tests
```
