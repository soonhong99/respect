# Environment

All results in the paper were measured on one Jetson AGX Thor Developer Kit.
Output tokens, KV caches and trajectories depend on the inputs, model weights,
software versions, kernels and sampling settings. Decode times additionally depend
on the power mode and clocks.

## Tested environment

| Item | Version |
|---|---|
| Board | Jetson AGX Thor Developer Kit (128 GB unified memory) |
| JetPack / L4T | 7.2 / R39.2 |
| GPU driver / CUDA runtime | 595.78 / 13.0 (driver supports 13.2) |
| Python | 3.12.13 |
| PyTorch | 2.8.0 built from source (commit `ba56102`), CUDA 13.0 |
| Triton | 3.7.1 |
| transformers | 4.57.1 |
| Alpamayo 1.5 | [NVlabs/alpamayo1.5](https://github.com/NVlabs/alpamayo1.5) at `2eff703` |
| Power mode | 120 W (`nvpmodel`), clocks fixed with `jetson_clocks`, CPU 2,601 MHz |

The complete package list is in [`environment/thor_pip_freeze.txt`](../environment/thor_pip_freeze.txt).

## Paper environment setup (PyTorch 2.8)

1. Flash JetPack 7.2 and select the 120 W power mode.
2. Build PyTorch v2.8.0 from source with CUDA 13.0 (see *PyTorch build* below).
3. Install Alpamayo 1.5 at the tested commit into the same environment,
   following its README (`uv sync`), and log in to Hugging Face to access the
   gated model and dataset.
4. Install this package without letting pip touch PyTorch:

   ```bash
   pip install --no-deps -e .
   pip install --no-deps einops
   ```

   Resolving dependencies automatically can replace the source-built PyTorch
   with a generic wheel, because a pre-release version such as
   `2.8.0a0+gitba56102` does not satisfy `torch>=2.8` for pip. Check with
   `pip install --dry-run` first.
5. Download the weights:

   ```bash
   hf download nvidia/Alpamayo-1.5-10B
   hf download z-lab/Alpamayo-1.5-10B-DFlash
   ```

6. Before timing runs, fix the clocks and check the environment:

   ```bash
   sudo jetson_clocks
   python scripts/collect_env.py
   ```

## PyTorch build

The measurements use PyTorch v2.8.0 built from source for Thor's GPU
(compute capability 11.0, `sm_110`) and CUDA 13.0, with the patch in
[`environment/pytorch-v2.8.0-thor.patch`](../environment/pytorch-v2.8.0-thor.patch).
The patch only makes the build work with CUDA 13 and Triton 3.7:

| Files | Change |
|---|---|
| `aten/src/ATen/cuda/cub.*`, `native/cuda/{Embedding,EmbeddingBag,Nonzero,TensorTopK,UniqueCub}.cu`, `test/cuda_cub_test.cu`, `cmake/Modules/FindCUB.cmake` | CCCL in CUDA 13 removed `cub::TransformInputIterator` and moved the CUB headers; use `thrust::make_transform_iterator` and the new include path |
| `native/cuda/CuFFTUtils.h` | cuFFT in CUDA 13 removed three error codes |
| `native/transformers/cuda/flash_attn/flash_api.cpp` | accept compute capability 11.x in FlashAttention's architecture check |
| `torch/utils/_triton.py`, `torch/_inductor/{async_compile,codecache}.py` | Triton 3.7 no longer exports `triton_key`; fall back to the Triton version |

```bash
git clone --branch v2.8.0 --recursive https://github.com/pytorch/pytorch && cd pytorch
git apply /path/to/respect/environment/pytorch-v2.8.0-thor.patch
export TORCH_CUDA_ARCH_LIST=11.0 USE_DISTRIBUTED=0 USE_CUSPARSELT=0 USE_FBGEMM=0
python setup.py develop
pip install --no-deps torchvision==0.23.0 triton==3.7.1
```

Use Triton 3.7.1. On this board, Triton 3.6.0 (the version bundled with the
PyTorch 2.11 wheel) failed to compile Triton kernels for `sm_110a` in `ptxas`.

## PyTorch 2.11 wheel

This is the installation route checked from a new isolated environment on Thor.
It uses the existing JetPack/CUDA installation, Python 3.12 runtime, and download
caches. The validation used `uv` 0.12.5; install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and Git first. These commands start in the directory where you want the checkout:

```bash
git clone https://github.com/soonhong99/respect.git
cd respect
uv venv --python 3.12 .venv
source .venv/bin/activate

# Official CUDA 13.0 wheels and their CUDA dependencies.
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130

# Pinned Python dependencies, including the tested Triton override.
uv pip install --no-deps -r environment/requirements-thor211.txt

git clone https://github.com/NVlabs/alpamayo1.5.git alpamayo1.5
git -C alpamayo1.5 checkout 2eff7037e47afb96a578b3d1bca453a373cd781e
uv pip install --no-deps -e ./alpamayo1.5
uv pip install --no-deps -e .

python -c "import torch,triton,transformers; print(torch.__version__,triton.__version__,transformers.__version__); print(torch.cuda.is_available())"
```

Expected versions are `2.11.0+cu130`, `3.7.1`, `4.57.1`, with CUDA available.
[`requirements-thor211.txt`](../environment/requirements-thor211.txt) pins the Python
packages; the complete earlier environment snapshot remains in
[`thor_torch211_freeze.txt`](../environment/thor_torch211_freeze.txt).

Keep the `--no-deps` steps above: the upstream Alpamayo metadata pins PyTorch 2.8,
so `uv sync` would select a different environment. The tested path uses SDPA and
does not install `flash-attn`. PyTorch 2.11 declares Triton 3.6.0, which is replaced
with the tested 3.7.1 for the Thor kernels. Consequently, `uv pip check` reports
those three dependency differences. It also reports a platform-tag incompatibility
for the installed NVIDIA `nvidia-cusparselt-cu13==0.8.0` SBSA wheel. The official
wheel installation, CUDA loading and two-frame checks succeeded with these four
reported incompatibilities; this validation does not resolve the SBSA tag warning.
A nonzero dependency check is therefore recorded, not described as a clean pass.

After accepting the gated model and dataset access conditions on Hugging Face:

```bash
hf auth login
hf download nvidia/Alpamayo-1.5-10B
hf download z-lab/Alpamayo-1.5-10B-DFlash
sudo jetson_clocks
python scripts/collect_env.py
python examples/decode.py --inputs runs/example_inputs --prepare-inputs-only
python examples/decode.py --inputs runs/example_inputs --out runs/decode_example.json
```

Existing Hugging Face login and cached downloads may be reused. The installation
check rebuilt two inputs and checked both frames against K, using a new Triton
compilation cache. It did not re-download all model/data bytes, reinstall JetPack
or CUDA, test Flow, or repeat the paper's performance evaluation.

With this wheel, the earlier 1,000-frame evaluation gave `#4` outputs bitwise identical
to `deterministic-kernel autoregressive decoding`. That autoregressive reference differed from
the PyTorch 2.8 paper reference on 19 frames. Use the PyTorch 2.8 setup for the
paper environment; the fresh two-frame installation check does not replace it.

## Other GPUs

The code contains no Thor-specific operator, but it has only been run on the
tested environment. The exact tokens and all timings in the paper are specific
to that environment.
