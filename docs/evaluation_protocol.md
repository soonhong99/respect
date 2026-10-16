# Evaluation protocol

This document details the evaluation conditions, timing boundaries, statistical resampling rules, and safety-evaluation assumptions used in Sections III, IV, and V of the paper.

[Method details](method_details.md) define the acceptance and output-preservation references, while [the data inventory](../data/README.md) explains clip selection, scored frames, and preceding history.

| Topic | Explanation |
|---|---|
| Evaluation and development data | [1. Data scope](#1-data-scope) |
| Offline candidates and execution delay | [2. Offline comparison](#2-offline-comparison) |
| Development-frame timing combination | [3. Setting selection](#3-setting-selection) |
| Included and excluded operations | [4. Decode timing boundary](#4-decode-timing-boundary) |
| Warmup, repetition and confidence intervals | [5. Repetition and statistics](#5-repetition-and-statistics) |
| Actual arrival, processing and publication times | [6. Real-time execution](#6-real-time-execution) |
| Calculated stopping positions and assumptions | [7. RSS-based evaluation](#7-rss-based-evaluation) |
| Ground-truth trajectory error and fixed noise | [8. Trajectory evaluation](#8-trajectory-evaluation) |
| Timing, event alignment and comparison environments | [9. Real-time result interpretation](#9-real-time-result-interpretation) |

## 1. Data scope

The offline evaluation scores 400 event-window frames from 10 clips and 600 nominal-driving frames from 60 clips of [NVIDIA PhysicalAI-Autonomous-Vehicles](references.md#models-and-data) (Section V-A).

The separate development sets score 600 event-window frames from 15 clips and 300 nominal-driving frames from 30 clips.

Preceding frames supply previous CoCs; they are separate from the scored frames used to report Decode performance.

The [selection rationale and frame counts](../data/README.md#selection-and-frame-scope) distinguish these scopes, including the adjacent-pair comparison of CoC reuse (Section III-A, Fig. 3(a)).

## 2. Offline comparison

The offline $d = 1$ condition assumes that the immediately preceding frame's completed CoC is available (Section III-A, Section V-B).

It gives the configurations the same inputs and previous CoCs, allowing their Decode costs to be compared under the same candidate history.

In execution, $d = 1$ corresponds to processing every frame at its 0.1 s arrival interval; the offline comparison does not establish that the full pipeline achieves that cycle time on Thor.

Both offline development and evaluation use previous-frame CoCs obtained by processing the same clip in advance with baseline autoregressive decoding, for the initial candidates and the drive index (Section V-A).

In real-time execution, RESPECT Decode instead uses only CoCs completed earlier in that same run (Section III-C, Section V-E).

Candidate differences affect acceptance length rather than the output determined by target verification; real-time runs separately evaluate reuse of the system's own outputs.

For configurations `#0`–`#4`, the public Prefill is executed once per frame and the resulting state is shared across Decode comparisons (Section V-A).

FlashDrive BF16 (`#0′`) uses its own Prefill and processes each clip's consecutive frames in order so its intended image-KV reuse condition holds.

All configurations use greedy decoding at temperature 0; FlashDrive's W4A8 quantization is disabled for the BF16 comparison, while its streaming, DFlash, and compilation optimizations are retained.

The main configurations use the PyTorch 2.8 environment, whereas `#0′` requires PyTorch 2.11.

`#4` is also measured in `#0′`'s PyTorch 2.11 environment, and direct Decode comparisons with `#0′` use that same-environment `#4` measurement (Section V-A); the [environment guide](environment.md) records both software setups.

## 3. Setting selection

Hyperparameters are selected on the development sets and applied to evaluation without re-tuning (Section IV-B, Section IV-C, Section V-A).

$N$ is compared with the continuation settings held at their selected values, and continuation settings are compared at $N = 4$.

Each continuation setting is measured on the 123 development frames requiring continuation at $N = 4$; frames completed by initial verification retain their measured times from the $N$ comparison.

Thus, a continuation-setting mean combines its measured continuation-frame times with the initial-only frame times from the $N$ comparison, within each development set.

The selection criterion is the average of the two development sets' mean Decode times, rather than a frame-count-weighted mean across the two sets.

The [setting-selection discussion](supplement.md#2-setting-selection-on-the-development-set) gives the 1% equivalence rule and the joint draft-tree size/top-$k$ choice.

The datastore's static index is built before execution from all 3,300 development CoCs, including preceding frames; its drive index receives completed CoCs from the same clip.

Static-index construction takes 35 ms and is outside the Decode timing boundary; inserting a new CoC into the drive index takes a median of 0.03 ms and is included.

## 4. Decode timing boundary

Alpamayo 1.5 runs in BF16 on Jetson AGX Thor, in the 120 W power mode with a fixed CPU clock of 2,601 MHz (Section V-A).

Decode time is the wall-clock duration of the Decode call with CUDA synchronization at both boundaries.

| Included in Decode time | Excluded from Decode time |
|---|---|
| Initial-trie construction and verification | Input loading |
| Accepted-path KV transfer into continuation verification | Prefill, including Vision Encoding |
| Continuation-candidate supply and verification | Writing the Prefill KV into the fixed buffer |
| FD-DFlash target-feature initialization | One-time static-index construction |
| Insertion of a newly available CoC into the drive index | Model-weight loading and Flow |

Since FlashDrive's public implementation does not delimit a Decode stage, its measured Decode interval runs from the end of its Prefill function to completion of the CoC, twice per frame like `#4`.

These Decode measurements are distinct from full-cycle wall-clock measurements in real-time execution.

## 5. Repetition and statistics

With clocks fixed, each configuration is warmed up once on the first frame (Section V-A).

Before each subsequent frame, `#3` and `#4` are each run once to exclude Triton compilation and initial memory allocation from the measurement.

FlashDrive BF16 is warmed up on the first three frames of the first clip, including compilation.

Each scored frame is timed twice, with the configurations executed in forward and reverse order, and the frame value is the average of those two timings.

Reported mean, p99, and observed maximum Decode times are computed over these frame values; repeated timings do not add unique frames.

| Comparison | Paper section | Resampling unit and count | Reported interval |
|---|---|---|---|
| Offline paired mean Decode-time difference | Sec. V-A / V-B | Clips, 2,000 resamples | 2.5th–97.5th percentiles, 95% interval |
| Real-time paired event-response difference | Sec. V-E | Clips, 10,000 resamples | 95% interval |
| Mean minADE6 comparison | Sec. V-G | Clips, 10,000 resamples | 95% interval |

Clip resampling accounts for dependence among consecutive frames in the same recording.

CoC preservation is assessed against `deterministic-kernel autoregressive decoding`, with committed-KV equality recorded separately; [method details](method_details.md#1-context-and-output-reference) distinguish this reference from public Prefill/Flow preservation.

## 6. Real-time execution

Inputs arrive every 0.1 s, each started processing cycle runs to completion, and after completion the latest waiting frame is processed while other waiting frames are dropped (Section III-C, Section V-A, Section V-E).

A started cycle completes because whether a newly arrived input contains an event is known only after processing it; processing all queued inputs would move the processed input timestamp further into the past.

For a run starting at frame $g_0$, recorded frame $g$ is supplied at $0.1(g - g_0)$ seconds after the run starts.

Each cycle runs input preparation, Prefill including Vision Encoding, Decode, and Flow for one trajectory, then publishes the trajectory.

Input preparation uses images from the last four recorded time steps, preprocessing, and tokenization.

This is open-loop replay of recorded drives: generated trajectories do not move the vehicle or change subsequent inputs, and the reported stopping positions are computed from replay.

The evaluation uses five starting frames per event clip, beginning 60 frames before the annotated event start $E$ and running to the clip end, giving 50 matched clip–start-frame pairs across configurations.

Initial candidates and the drive index contain only CoCs from processing completed earlier in the same run ($\mathcal{H}_t = \{Y_j \mid j < t,\, C_j \le S_t\}$ in Section III-C).

The delay $d$ is the input-frame index difference between the current cycle and the preceding completed cycle whose CoC is available; its distribution excludes the last cycle, whose final interval is constrained by the end of the clip.

A cycle's duration spans input preparation through trajectory publication; event response latency additionally reflects when the event arrives during processing (Section III-C, Section V-E).

| Metric | Definition |
|---|---|
| Trajectory age | Elapsed time since the input timestamp of the trajectory currently in use |
| Trajectory expiry | Fraction of measured time when that age exceeds the trajectory's 6.4 s prediction horizon |
| Event response latency | Time from arrival of annotated event-start frame $E$ until publication of the first trajectory generated from an input at or after $E$ |

A trajectory first appears after input waiting and processing, then remains in use until the next publication, so its age ranges approximately from one to two cycle times.

## 7. RSS-based evaluation

Measured event response latency is substituted into the [RSS longitudinal minimum safe distance model](references.md#rss) (Section V-F, Eq. (1)):

$$
d_{\min}(\rho)=v_r\rho+\frac{v_r^2}{2b_{\min}}-\frac{\max(v_f,0)^2}{2b_{\max}}.
$$

At event-start frame $E$, $s_0$ is the gap to the nearest lead object in the ego lane, $v_r$ is ego speed, $v_f$ is the lead object's speed projected along the ego heading, and $\rho$ is response time.

Ego motion and automatically generated object labels at $E$ supply the speeds, lead-object type, and position; event timing and type follow the dataset's OOD annotation.

The minimum ego braking deceleration $b_{\min} = 4\text{ m/s}^2$ and maximum lead-object braking deceleration $b_{\max} = 8\text{ m/s}^2$ follow a calibration example based on naturalistic driving data ([Xu et al.; Hasuo](references.md#rss)).

The allowable response time $\rho^*$ satisfies $d_{\min}(\rho^*) = s_0$.

Setting $\rho$ to measured event response latency assumes the ego vehicle maintains its event-start speed until the new trajectory appears and then brakes at $4\text{ m/s}^2$.

For a moving lead object, the reference is its stopping position under maximum braking deceleration ($8\text{ m/s}^2$).

The signed stopping margin is $v_r(\rho^* - \rho)$, the signed separation between the computed stopping positions: positive means stopping before the lead object's stopping position, and negative means passing it.

Of ten event clips, six had a lead object in the ego lane at an ego speed of at least $0.5\text{ m/s}$.

In two events, every run's event response latency was within $\rho^*$ ($7.8$ and $34.4\text{ s}$); in the remaining four, every repetition of `#0` and `#4` had a latency exceeding $\rho^*$ ($1.28\text{–}2.91\text{ s}$).

[Figure 12](../figures/fig12_stopping_margin.png) shows those four events, with values averaged over the five start frames.

In the $72\text{ km/h}$ truck event, $\rho^*$ was $1.28\text{ s}$, below even the shortest measured RESPECT cycle ($1.50\text{ s}$), yielding a $-65\text{ m}$ mean stopping margin under RESPECT.

These computed replay margins depend on the stated constant-speed and braking assumptions; they are not observed closed-loop stopping outcomes.

## 8. Trajectory evaluation

Minimum average displacement error over six trajectories (`minADE6`) is the smallest mean position error against 64 recorded future positions ($6.4\text{ s}$), computed across six trajectories generated with fixed noise seeds (Section V-G).

The compared configurations are the public implementation (`#0`), RESPECT, and FlashDrive BF16 (`#0′`).

For identical inputs, RESPECT's accelerated Prefill and Flow preserve the public stages' outputs bit for bit, so RESPECT's trajectory error is evaluated using `#4`, which combines public Prefill/Flow with RESPECT Decode.

Each configuration generates six trajectories per frame with six distinct noise samples held fixed across frames.

`#4` matches `deterministic-kernel autoregressive decoding` in CoC and committed-path KV on all 1,000 frames; that reference matches `#0`'s CoC on 975 frames and differs on 25.

Per-frame absolute `minADE6` differences and differences between group means are reported separately: among the 25 different-CoC frames, the difference between mean `minADE6` values of RESPECT and `#0` is below $0.05\text{ m}$, while the largest per-frame absolute differences are $0.55\text{ m}$ (event-window) and $0.13\text{ m}$ (nominal-driving).

The [detailed results](../README.md#detailed-results) summarize the main metrics, and the [source map](../REPRODUCE.md#paper-item--source) links each number to its raw file and JSON field.

## 9. Real-time result interpretation

Replacing only baseline Decode with RESPECT Decode (`#4`) shortened processing time in all 50 matched clip–start-frame pairs, but `#4` responded later than `#0` in 11 of 50 pairs (Section V-E).

In those cases, `#4` started processing an earlier input just before event onset $E$, so the event frame had to wait until that cycle finished.

This arrival alignment explains why a shorter cycle time reduces mean response latency without reducing it on every individual pair.

After the runs, all 403 frames processed by `#4` were recomputed with `deterministic-kernel autoregressive decoding`, and every CoC token sequence matched.

In the PyTorch 2.8 `#0`/`#4` environment, the `#4` cycle and mean event response latency were $2.39\text{ s}$ and $3.81\text{ s}$; in the PyTorch 2.11 FlashDrive/RESPECT comparison environment, they were $2.38\text{ s}$ and $3.80\text{ s}$.

Those are separate measurements, and comparisons with FlashDrive and RESPECT (`#5`) use the `#4` run from their shared PyTorch 2.11 environment.

In that environment, adding RESPECT's bit-exact Prefill/Flow acceleration lowers median $d$ from 23 to 16 and raises cycles completed by initial verification from 23.3% to 30.8%, reducing Decode time as well as the other stages.

The [supplement](supplement.md#8-flashdrive-stage-comparison-and-output-differences) separately describes the FlashDrive + RESPECT Decode run, which uses FlashDrive's Prefill and Flow rather than RESPECT's Prefill/Flow acceleration.
