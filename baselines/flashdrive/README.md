# FlashDrive BF16 baseline (#0′)

#0′ is [FlashDrive](https://github.com/z-lab/flashdrive) at commit
`08fc5d062fc1ed74ce2da916de51256dc8bcf4e9`, unmodified except that the lm_head row of
its added mask token is set to zero (below), run with the z-lab Alpamayo 1.5
checkpoint. Its W4A8 quantization is disabled so that it runs the same BF16
target as RESPECT; streaming, DFlash, expert fusion and `torch.compile` stay
enabled. CoCs are decoded greedily (temperature 0).

## Setup

Check out the tested commit; it is added to `sys.path`, not installed:

```bash
git clone https://github.com/z-lab/flashdrive && cd flashdrive
git checkout 08fc5d062fc1ed74ce2da916de51256dc8bcf4e9
```

The scripts run in the PyTorch 2.11 wheel setup of
[docs/environment.md](../../docs/environment.md), which FlashDrive supports.
FlashDrive's pure-Python dependencies can be added with `--extra-path`
(for example the `site-packages` of FlashDrive's own environment). On Jetson AGX
Thor, point Triton to the system `ptxas`:

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
```

## #0′ on the evaluation frames (Section V)

The evaluation inputs hold only the scored frames, and FlashDrive's streaming
Prefill needs a full first window before them. `fill_windows.py` builds the
window of the frame before each clip's first scored frame with the same dataset
loader (CPU; needs the dataset) and checks its overlap with the first scored
input. `offline_unified.py` then loads the public model and FlashDrive in one
process and, for every clip, starts a new stream with that window and processes
the scored frames in order (d = 1, so the previous window's KV reuse holds). Each
clip is processed twice. Decode is timed from the end of FlashDrive's Prefill to
the end of its generation (first token, draft/verify loop, trajectory-start
forward), the boundary of #4. The same process also runs #4 on the same frames
and checks the output-preserving Prefill and Flow of `respect/system` against
the public implementation:

```bash
python baselines/flashdrive/fill_windows.py --inputs INPUTS --out FILL_WINDOWS
python baselines/flashdrive/offline_unified.py --inputs INPUTS --fill FILL_WINDOWS \
    --flashdrive /path/to/flashdrive --out runs/flashdrive_unified [--smoke]
```

One process holding both models reuses FlashDrive's stream state across clips,
which FlashDrive itself does not do: the script clears the KV cache and the
shape-specific vision caches before each first window and restores the
4-image-window caches afterwards, so the compiled graphs keep reading the same
objects.

## minADE6 of #0′ (Section V-G)

`quality_fd.py` runs #0′ as `offline_unified.py` does, with six trajectory samples
instead of one, and records the CoC tokens and the minADE6 / minFDE6 of every scored
frame. With `--fixed-seed 0` the Flow noise is the same on every frame, as for #0 and
#4; the paper uses this run (`results/paper/flashdrive_quality/`).

```bash
python baselines/flashdrive/quality_fd.py --inputs INPUTS --fill FILL_WINDOWS \
    --flashdrive /path/to/flashdrive --fixed-seed 0 --out runs/flashdrive_quality
```

## Stages linked with RESPECT Decode in real time (Section V-E)

FlashDrive's streaming Prefill reuses the previous window's KV and computes only
the newest frame, which requires consecutive windows 0.1 s apart. With the
latest-frame policy of the real-time runs every cycle prefills the whole 16-image
window through FlashDrive's compiled path. `stage_link.py` loads the public model
and FlashDrive in one process and runs, with the schedule of
`scripts/run_realtime.py`, FlashDrive BF16 as published (`FC8`), FlashDrive with
RESPECT Decode (`D8`), the #4 run (`S4`) and five variants that swap the Flow
steps or weights. FlashDrive keeps keys before RoPE and the public implementation
after RoPE, so keys are rotated with the same position table when KV moves
between them. FlashDrive's `setup_dflash` leaves the lm_head row of its added mask
token at its random initial value; the script sets that row to zero so that only
real vocabulary tokens are chosen. With both models loaded the process peaks at
about 57 GiB of GPU memory. FlashDrive is added to `sys.path`, not installed:

```bash
python /path/to/respect/baselines/flashdrive/stage_link.py --inputs REALTIME_INPUTS \
    --flashdrive /path/to/flashdrive --out runs/realtime_flashdrive [--smoke]
```

`REALTIME_INPUTS` is the output of `scripts/build_realtime_inputs.py`. The
checkpoint `z-lab/Alpamayo-1.5-10B` has the same language-model and vision
weights as `nvidia/Alpamayo-1.5-10B` but different Flow weights, so its
trajectories differ from the public implementation's at equal CoC.

FlashDrive's `torch.compile` max-autotune may pick different kernels in each
session, so FlashDrive trajectories can differ by millimetres to centimetres
between sessions; the token checks of `--smoke` do not depend on this.
