# Reproducing the paper

The public raw results are in [`results/paper`](results/paper). The commands below
recompute their summary statistics and generate all figures without a GPU.
The source map below identifies the exact file and field for every reported claim
(note: the individual per-frame log of the earlier development pairwise-kernel run
behind the `3/400` count in Section IV-D is not included in `results/paper`,
Figures 2, 5, and 6 are conceptual/method illustrations, and Fig. 13 is rendered
as BEV without the gated camera panel).

## Find the commands, conditions, and evidence

| Need | Location |
|---|---|
| Recompute saved summaries and figures without inference | [Commands below](#recompute-the-saved-results) |
| Run GPU measurements | [Runs and commands](#runs), [tested environment](docs/environment.md) |
| Understand the verification algorithm and preservation references | [Method details](docs/method_details.md) |
| Check data scope, timing boundaries, and evaluation assumptions | [Evaluation protocol](docs/evaluation_protocol.md), [data selection](data/README.md#selection-and-frame-scope) |
| Find the raw file and field supporting a paper result | [Paper item → source](#paper-item--source), [raw-result inventory](results/paper/README.md) |
| Read the `3/400` pairwise-kernel comparison and its scope | [Kernel reduction-order comparison](docs/method_details.md#8-kernel-reduction-order-comparison) |
| Read the auxiliary Flow-weight swap check and its scope | [Stage comparison](docs/supplement.md#8-flashdrive-stage-comparison-and-output-differences) |
| Locate the background sources cited in the documentation | [References](docs/references.md) |
| Browse the index of all paper figures, tables, and equations | [Paper figures, tables, and equations](docs/manuscript_map.md) |

## Recompute the saved results

```bash
python scripts/summarize.py results/paper          # writes results/paper/summary.json and prints the headline numbers
python scripts/summarize_additions.py results/paper # writes results/paper/summary_additions.json (observed delays, minADE6 of #0', N under delay)
python scripts/make_figures.py results/paper --out figures   # measurement figures, supplement figures and method illustrations (Fig. 13 BEV only)
```

To regenerate just the two system-level comparisons, using only the public JSON
files and the same NumPy/Matplotlib dependencies:

```bash
python scripts/paper_system_figures.py results/paper --out figures
```

- **Fig. 11** (`fig11_realtime_breakdown.png`): `summary.json`,
  `section6.realtime.configs.{S0,S4}` and
  `section6.stage_link.configs.{S4F,FC8,D8}`. Input + Prefill combines
  `input_ms` and `prefill_ms`; the stage bars show mean times per processed cycle,
  and the response panel shows the mean and observed maximum over 50 runs.
  `#0`/`#4` use PyTorch 2.8, and the other rows use PyTorch 2.11, as in the paper.
  The groups indicate retained or changed model weights, not equality with
  the public pipeline. Decode preservation uses the `deterministic-kernel autoregressive decoding`
  reference (`K`); Prefill/Flow preservation uses the public stage on identical inputs.
- **Fig. 13** (`fig13_trajectories.png`):
  `frame_trajectories/bbbe775f_0160.json`, six 64-position xyz trajectories per
  system and the recorded path. Maximum coordinate difference and mean-endpoint
  distance are calculated from those arrays, not entered as plot constants.
  This public rendering includes the BEV comparison from the paper's event frame;
  it omits the front-camera panel because sensor images are not redistributed.
  Coordinates use $x$ forward and $y$ left; the horizontal plot coordinate is $-y$ so
  that the vehicle's left appears to the left. The RESPECT curves use the `#4`
  measurements, as in the paper; this is not an exact-match claim against `#0`.

To rerun the measurements, prepare the environment ([docs/environment.md](docs/environment.md))
and the input files (`<clip_id>_<frame:04d>.pt`, built from the dataset with the
`clip_id` and `t0_us` listed in [`data/`](data)), then run the scripts below on a
Jetson AGX Thor with the clocks fixed (`sudo jetson_clocks`, 120 W).

## First Decode run

For a small functional check before the evaluation suite, follow the
[Minimal Decode example](README.md#minimal-decode-example). It includes CPU input
preparation and a two-frame RESPECT Decode (`#4`) stream with token and committed-KV
checks against `K`. It starts with empty history and does not measure latency, run
Flow, or replace the evaluation protocol below.

## Protocol

The [evaluation protocol](docs/evaluation_protocol.md) gives the paper's timing boundaries, warmup, sample units, offline versus real-time candidate history, and RSS/trajectory assumptions.

The [data inventory](data/README.md#selection-and-frame-scope) distinguishes scored frames, preceding history, and adjacent-frame pairs.

Hyperparameters are fixed on a separate development set before evaluation. In the
main offline comparison, each configuration is timed twice per evaluated frame,
in forward and reverse configuration order; each frame contributes the mean of
those two timings. The repeated timings are not additional unique frames.
FlashDrive uses two clip passes, as described in the `#0′` note below.

1. **Development set** (`data/dev_{event,general}.jsonl`): 15 event-window clips (at most 5
   per event type, drawn from the event-window candidates that are not evaluated) and
   30 nominal-driving clips (the clips after the 60 evaluated ones in the same
   random order; denoted `general` in filenames). Selection does not look at model outputs.
2. **Selection** (`select_hparams.py`, Sections IV-B and IV-C): $N$, the FD-DFlash tree size and top-$k$, the
   continuation node budget, and the datastore scope and key length, by a rule fixed before the
   run (smallest mean Decode time; within 1%, the smallest parameter value).
3. **Evaluation** (`run_main.py --flow`, Section V): all configurations, the delay
   comparison, and the stage times of the public implementation, with the fixed
   settings and timing protocol above. The
   datastore's static index holds only development CoCs; the evaluated frames
   contribute only the CoCs already produced earlier in the same clip.
4. **Execution delay and real-time runs** (Sections V-D to V-F): the delay comparison
   extended to $d = 30, 40$; the real-time runs on the board with the latest-frame
   policy; the RSS evaluation; FlashDrive's stages linked with RESPECT Decode and the
   bit-exact Prefill and Flow, both in real time (PyTorch 2.11 wheel setup,
   [docs/environment.md](docs/environment.md)). The same hyperparameter values from step 2 are used,
   unchanged.
5. **Observed-delay and trajectory comparisons** (Sections V-D and V-G): the delay comparison at
   the median delays observed in the real-time runs ($d = 16, 23$, with $d = 20$ again),
   with earlier frames of the nominal-driving clips (`general`) so that frames $t - d, t - 2d, \dots$ exist as on
   the event-window clips, and the minADE6 of FlashDrive BF16 (`#0′`) with six trajectories
   and the same Flow noise on every frame.

## Runs

| Run | Command | Thor time | Output |
|---|---|---|---|
| Development set: clips, reference CoCs, inputs | `python scripts/collect_dev.py --phase select --snapshot SNAPSHOT --out runs/dev_set`, then `--phase collect --inputs DEV_INPUTS` | 3 h (after downloading the camera data of the event clips) | `runs/dev_set/dev_{event,general}.jsonl` (= `data/dev_*.jsonl`) |
| Hyperparameter selection | `python scripts/select_hparams.py --inputs DEV_INPUTS --out runs/dev` | 3 h | `runs/dev/frames_{event,general}.jsonl` ($N$ sweep), `followup_{event,general}.jsonl` (continuation settings at the chosen $N$), `selection.json` |
| Query-tile boundary | `python scripts/run_tile_boundary.py --inputs DEV_INPUTS --out runs/dev/tile_boundary.json` | 5 min | `runs/dev/tile_boundary.json` |
| Main evaluation: ten configurations, delay comparison, trajectories, stage times, power, memory | `python scripts/run_main.py --inputs INPUTS --out runs/main --flow` | 6.5 h | `runs/main/frames_{event,general}.jsonl` |
| Kernel summation order | `python scripts/run_kernel_order.py --inputs INPUTS` | 1 min | printed |
| Delay $d = 30, 40$ | `python scripts/run_main.py --inputs INPUTS --out runs/delay --delay-only --delays 20,30,40` | 1 h | `runs/delay/frames_{event,general}.jsonl` |
| Nominal-driving set (`general`), $d = 40$: earlier frames and the delay run | `python scripts/collect_history.py --set general --delays 40 --out data/general_history_d40.jsonl`, then `python scripts/run_main.py --inputs INPUTS --out runs/delay_d40 --sets general --delay-only --delays 40 --modes stride --history data/general_history_d40.jsonl` | 1.3 h + 0.5 h | `data/general_history_d40.jsonl` (1,140 frames), `40_stride` of `runs/delay_d40/frames_general.jsonl` (merged into `results/paper/delay/`) |
| Nominal-driving set (`general`), $d = 16, 20, 23$: earlier frames and the delay run | `python scripts/collect_history.py --data DATA --set general --delays 16,20,23 --out data/general_history_d16_d23.jsonl` with `DATA` = a copy of `data/` whose `general.jsonl` also lists `data/general_history_d40.jsonl` (those frames are reused), then `python scripts/run_main.py --inputs INPUTS --out runs/delay_observed --sets general --delay-only --delays 16,20,23 --modes stride --history HISTORY` with `HISTORY` = `general_history_d40.jsonl` followed by `general_history_d16_d23.jsonl` | 2.7 h + 0.6 h | `data/general_history_d16_d23.jsonl` (2,327 frames), `runs/delay_observed/frames_general.jsonl` |
| Event-window set (`event`), $d = 16, 20, 23$ | `python scripts/run_main.py --inputs INPUTS --out runs/delay_observed --sets event --delay-only --delays 16,20,23 --modes stride` | 0.5 h | `runs/delay_observed/frames_event.jsonl` |
| minADE6 of FlashDrive BF16 (`#0′`), six trajectories, same Flow noise on every frame (Sec. V-G) | `python baselines/flashdrive/quality_fd.py --inputs INPUTS --fill FILL_WINDOWS --flashdrive FLASHDRIVE --fixed-seed 0 --out runs/flashdrive_quality` (PyTorch 2.11 setup, [baselines/flashdrive](baselines/flashdrive)) | 32 min | `runs/flashdrive_quality/fd_quality.jsonl` (trajectories removed in `results/paper` except the frame of Fig. 13) |
| Component comparison on the development set ([Supplement §3](docs/supplement.md#3-ablations)) | `python scripts/run_main.py --inputs DEV_INPUTS --out runs/dev_ablation --sets dev_event,dev_general` | 2.5 h | `runs/dev_ablation/frames_dev_{event,general}.jsonl` |
| Real-time inputs (event clips to the end of the recording) | `python scripts/build_realtime_inputs.py --out REALTIME_INPUTS --check-inputs INPUTS` | 16 min (CPU) | `REALTIME_INPUTS/<clip>/` |
| Real-time runs | `python scripts/run_realtime.py --inputs REALTIME_INPUTS --out runs/realtime` | 1 h | `runs/realtime/{cycles,verify}.jsonl` |
| Lead objects for the RSS evaluation | `python scripts/extract_rss_leads.py --out runs/realtime/rss_leads.json` | 1 min (CPU) | `rss_leads.json` |
| FlashDrive stages linked with RESPECT Decode, real time (Sec. V-E) | see [baselines/flashdrive](baselines/flashdrive) (`stage_link.py`) | 2.5 h | `runs/realtime_flashdrive/{cycles,verify}.jsonl` |
| RESPECT (`#5`): RESPECT Decode with the bit-exact Prefill and Flow, real time (Sec. V-E) | `python scripts/run_realtime_fast.py --inputs REALTIME_INPUTS --out runs/realtime_fast` | 30 min | `runs/realtime_fast/{cycles,verify}.jsonl` |
| First windows for `#0′` (frame before the first scored frame) | `python baselines/flashdrive/fill_windows.py --inputs INPUTS --out FILL_WINDOWS` | 35 min (CPU) | `FILL_WINDOWS/*.pt`, `check.json` |
| FlashDrive BF16 baseline (`#0′`), `#4` in the same environment, bit-exact Prefill/Flow check | see [baselines/flashdrive](baselines/flashdrive) (`offline_unified.py`) | 2 h | `runs/flashdrive_unified/{fd,respect}.jsonl` |

Copy or link the outputs into one directory with the layout of `results/paper`
(`main/`, `dev/`, `flashdrive_unified/`, `delay/`, `dev_ablation/`, `realtime/`,
`realtime_flashdrive/`, `realtime_fast/`, `delay_observed/`, `flashdrive_quality/`) and run
`scripts/summarize.py` and `scripts/summarize_additions.py` on it. The development-set comparison files keep the names
`frames_{event,general}.jsonl` under `dev_ablation/`.

## Paper item → source

Section and figure numbers below match the RESPECT paper (`Secs. I–VI`, `Figs. 1–14`). See [Paper figures, tables, and equations](docs/manuscript_map.md) for the complete visual index, and [docs/supplement.md](docs/supplement.md) for the `supp_*` supplementary plots and tables.

Configuration codes: `S0` = `#0` (baseline autoregressive), `S1` = `#1`, `S2b` = `#2`, `S2a` = `#2′`, `S3` = `#3`, `S4` = `#4` (RESPECT Decode), `K` = `deterministic-kernel autoregressive decoding` (the output-preservation reference), `S1fK` / `S2bK` / `S2aK` = `#4` without both sources / without datastore / without FD-DFlash, `S4F` (`realtime_fast/`) and `S5` (`memory/`) = `#5` (RESPECT: RESPECT Decode with bit-exact Prefill and Flow), `FC8` = FlashDrive BF16, and `D8` = FlashDrive with RESPECT Decode in the real-time runs. The paper's `event-window` and `nominal-driving` sets are keyed as `event` and `general` in the data and result files.

Fields without a mark come from `summary.json` (`scripts/summarize.py`); fields marked `+` come from `summary_additions.json` (`scripts/summarize_additions.py`).

| Paper item | Run | Field | Paper value (event window / nominal driving) |
|---|---|---|---|
| Abstract, Sec. I, Sec. III-A, Fig. 3(a) (`fig3_concentration.png`): identical complete CoC of adjacent frames | — (data only) | `adjacent_coc` | 1,100/1,390 (79.1 %), 2,096/2,340 (89.6 %) |
| Sec. I, Sec. III-A, Figs. 2 and 3(b) (`fig3_concentration.png`): frames completed by initial verification, and the rest | main | `sets.*.configs.S4.initial_only_frames`, `sets.event.concentration` | 313/400 (78.3 %), 555/600 (92.5 %); the other 87 event-window frames (21.7 %) take 81.1 % of total Decode time |
| Sec. I, Sec. V-B, Fig. 8: continuation sources | main | `sets.*.sources` | joint vs. neither source −66.6 / −67.9 % mean on the 87 / 45 frames; datastore alone faster on 74 / 35 frames (`datastore_faster_frames`); observed max `#4` 2,033 / 1,702 ms vs. 2,481 / 1,748 (FD-DFlash only) and 2,340 / 2,712 (datastore only); oracle in `on_*_faster` |
| Sec. I, Sec. V-B: CoC different from `#0` | main + `#0′` | `sets.*.configs.S3.match_S0`, `sets.*.flashdrive.coc_match_default_sequential` | `#3`: 17 + 7 = 24 frames (383 / 593 equal); `#0′`: 192 + 121 = 313 frames |
| Sec. I, Sec. V-E: FlashDrive Flow on frames with the same CoC | realtime_flashdrive | `section6.stage_link.trajectory_same_coc` | per-frame maximum trajectory-position difference from `#0`: median 1.48 m (102 frames) |
| Fig. 1, Sec. V-F, Fig. 12(b): pedestrian-crowd event | realtime + realtime_fast | + `stopping_margin.bbbe775f` | ego 17.7 km/h; mean event response latency `#0` 5.52 s, RESPECT 2.55 s (3.0 s earlier); stopping margin −12.8 → +1.8 m (−13 → +2 m) |
| Sec. III-C, Fig. 4 (`fig4_realtime_timeline.png`) | realtime | `realtime/cycles.jsonl` (clip `61d3622b`, start phase 24) | event response latency 5.67 s (`#0` run) and 3.59 s (`#4` run), 2.08 s shorter |
| Sec. III-C, Sec. V-F: shortest RESPECT cycle | realtime_fast | minimum `cycle_ms` of `realtime_fast/cycles.jsonl` | 1.50 s (1,503 ms over 351 cycles) |
| Sec. IV-B, [Supplement §2](docs/supplement.md#2-setting-selection-on-the-development-set), `supp_n_select`: $N$ on the development sets | selection | `dev.nselect`, `dev.chosen` | completed 81.1 → 87.2 % ($N = 1 \to 4$); shortest mean at $N = 6$, $N = 4\text{–}12$ within 1 %; chosen $N = 4$ |
| Sec. IV-C, [Supplement §2](docs/supplement.md#2-setting-selection-on-the-development-set): node budget, tree size, top-$k$, key length | selection | `dev.one_at_a_time` | budget 64: 124.7 ms; budget 128: 127.9 ms (more tokens per run, 3.86 vs. 3.75) |
| Sec. IV-C, [Supplement §2](docs/supplement.md#2-setting-selection-on-the-development-set), `supp_tile_boundary` | tile boundary | `dev.tile_boundary` | 64 → 65 nodes: +14.3 ms per forward pass (median of the four frames) |
| Sec. IV-D, Fig. 6: 26 values in contiguous or scattered slots | kernel order | printed | `2.4414e-04` (pairwise), `0` (logical order) |
| Sec. IV-D: CoC different from `K` on the 400 event-window frames | main; the pairwise-reduction count comes from a run of the earlier development kernel ([Method details §8](docs/method_details.md#8-kernel-reduction-order-comparison)) | `sets.event.integrity` (logical-order accumulation, `#4`) | 0 frames (logical-order accumulation: tokens and KV equal on 400/400); 3 frames (pairwise reduction) |
| Sec. IV-E, Fig. 11: Prefill and Flow in a `#4` cycle | realtime | `section6.realtime.configs.S4.stage_mean_ms` | Prefill 1,388 ms, Flow 664 ms, 85.7 % of the 2.39 s cycle |
| Sec. IV-E: bit-exact Prefill and Flow on the 1,000 evaluation frames | `#0′` run | `flashdrive_same_environment.*.preserve` | Prefill KV, first token, hidden states, and trajectory equal to the public stages for the same input: 400/400, 600/600 |
| Sec. V-A: `#4` in the `#0′` environment (PyTorch 2.11) | `#0′` run | `flashdrive_same_environment.*.s4_mean_ms` vs. `sets.*.configs.S4` | 148.9 vs. 153.3 ms (−2.9 %), 101.4 vs. 101.0 ms (+0.4 %) |
| Sec. V-B, Fig. 7: `#1` vs. `#0` | main | `sets.*.configs.{S0,S1}` | −74.7 / −83.3 % mean; `#1` max 3,583 ms vs. `#0` 3,560 ms (nominal driving) |
| Abstract, Sec. V-B, Fig. 7: `#4` vs. `K` | main | `sets.*.configs.{S4,K}.latency_ms` | −88.7 / −91.3 % mean, −50.5 / −53.1 % max |
| Sec. V-B, Fig. 7: `#4` vs. `#0′` (same environment) | main + `#0′` | `sets.*.configs.S4`, `sets.*.flashdrive` | mean Decode time 64.7 / 66.5 % lower |
| Abstract, Sec. V-B, Sec. V-D: output preservation | main + delay | `sets.*.configs.S4.match_K`, `sets.*.integrity`, `sets.*.delay.*.mismatch` | CoC equal to `K` on 400/400, 600/600; 0 mismatches at every tested $d$ |
| Sec. V-B: deterministic kernels, `#4` vs. `#3` | main | `sets.*.kernel_percall_ms` (`supp_kernel_cost`); mean `rounds` (forward and reverse) of `S3` and `S4` in `main/frames_*.jsonl` on the 87 / 45 frames where `#4` needs continuation | each call 25–29 % shorter; 3.64 vs. 3.69 and 2.96 vs. 2.98 continuation runs per frame (at most 1.4 % difference) |
| Sec. V-C, Fig. 9: Decode time around the event start | main + `#0′` | `sets.event.event_timeline` (`seq` = `#0`, `fd` = `#0′`, `res` = `#4`) | `#4` window mean below 200 ms in every window; event onset (0–4): mean and max 90.8 % and 79.3 % below `#0`; `#0′` window mean highest in the three windows −5 to 9 (529–544 ms) |
| Abstract, Sec. V-D, Fig. 10: delay $d$ | main + delay + delay_observed | `sets.*.delay`, `section6.delay_extension.*.rows`; + `observed_delay.*_{16,23}_stride` | at $d = 16$ and $23$: −75.8 / −85.8 % and −74.3 / −85.2 % mean vs. `K` (at least 74 / 85 %); 0 mismatches at every tested $d$ |
| Fig. 10 shading, Sec. V-D, Sec. V-E: delay $d$ of the real-time runs | realtime + realtime_fast | + `realtime_delay` | median (10th–90th percentile) 16 (15–19) RESPECT run, 23 (22–25) `#4` run |
| Abstract, Sec. V-E, Fig. 11: `#0` and `#4` runs | realtime | `section6.realtime` | cycle 3.28 / 2.39 s; event response latency 5.34 / 3.81 s; `#0 − #4` 1.53 s [1.29, 1.75]; trajectory expiry 9.70 / 0.86 %; 0 mismatches over 403 frames |
| Abstract, Sec. V-E, Fig. 11: FlashDrive BF16, FlashDrive + RESPECT Decode, RESPECT | realtime_flashdrive + realtime_fast | `section6.stage_link.configs` (`FC8`, `D8`, `S4F`) | cycle 1.76 / 1.71 / 1.73 s; event response latency 2.83 / 2.86 / 2.84 s |
| Sec. V-E: paired differences | same | `section6.stage_link.pairs` | `#4 − RESPECT` 0.96 s [0.81, 1.12], 49/50; `FlashDrive − RESPECT` −0.01 s [−0.20, 0.21] |
| Sec. V-E: FlashDrive with RESPECT Decode | same | `section6.stage_link.configs.{FC8,D8}.stage_mean_ms.decode_ms`, `.D8.initial_complete_share` | Decode 352 → 306 ms (−13 %); 29.5 % of cycles completed by initial verification |
| Sec. V-E: 295 frames processed by RESPECT | same | `section6.stage_link.verify` | 295/295 equal, bit for bit, to a recomputation with public Prefill, `K`, and public Flow |
| Sec. V-E: `#4` run later than `#0` run | realtime | `section6.realtime.paired_S0_minus_S4.reaction_s.s4_shorter` | 39 of 50 pairs shorter, so 11 of 50 later |
| Sec. V-F, Eq. (1), Fig. 12: four RSS-critical events | realtime + realtime_fast + leads | + `stopping_margin` (`margin_m`, `rho_star_s`) | allowable response time 1.28–2.91 s; RESPECT − `#0` margin +19.9 (construction zone), +14.6 (pedestrian crowd), +23.1 (vehicle, 29 km/h), +35.5 m (truck, 72 km/h) = 15–36 m |
| Sec. V-F: the two events every run met | realtime | `section6.rss` | allowable response time 7.8 and 34.4 s |
| Sec. V-F: truck event | realtime_fast | + `stopping_margin.25c50d83`; minimum `cycle_ms` of `realtime_fast/cycles.jsonl` | $\rho^*$ 1.28 s, below the shortest RESPECT cycle of 1.50 s; RESPECT mean latency 4.54 s |
| Sec. V-G, Fig. 13: the three systems on one frame | frame_trajectories | + `fig13_frame` | RESPECT vs. `#0`: max. coordinate difference 0.0204 m (0.0244 m as 2-D distance); mean endpoints of `#0′` and `#0` 15.18 m apart; recorded stop 10.2 m ahead |
| Sec. V-G, Fig. 14: minADE6 of RESPECT (`#4`) vs. `#0` | main | + `trajectory_error.*.diff_from_0.#4.same_coc`, `trajectory_error.*.coc_diff_frames` | `K` gives the CoC of `#0` on 380 + 595 = 975 frames; there median 0.0022 / 0.0023 m; on the other 20 + 5 = 25 frames mean minADE6 difference below 0.05 m |
| Sec. V-G: mean minADE6 | main + flashdrive_quality | + `trajectory_error.*.mean_minADE6_m`, `fd_minus_4` | RESPECT vs. `#0` below 0.003 m; `#0′` 0.29 m [0.03, 0.56] / 0.38 m [0.22, 0.54] below RESPECT |
| Sec. V-G, Fig. 14: `#0′` vs. `#0` | flashdrive_quality | + `trajectory_error.*.diff_from_0.#0'` | absolute minADE6 difference median 0.43 / 0.33 m, max 5.19 m (nominal driving) |
| Sec. V-G: Decode-region energy | main | `sets.*.configs.{S0,K,S4}.energy_j_mean` | `#0` 95.56 / 80.59 J, `K` 100.88 / 84.98 J, `#4` 11.48 / 7.05 J (88.0 / 91.3 % below `#0`) |
| Sec. V-G: peak memory | memory | `memory/{S0,S4,S5}_summary.json` (`peak_alloc_mib`) | `#0` 21.61, `#4` 23.05 (+1.44 GiB, 6.7 %), `#5` 24.07 GiB (11.4 %) |

The related-method comparison is available in [Supplement §6](docs/supplement.md#6-comparison-with-related-acceleration-methods). The development-set component comparison, the query-tile boundary, and the per-call kernel cost are in [docs/supplement.md](docs/supplement.md) (`section6.dev_ablation`, `dev.tile_boundary`, `sets.*.kernel_percall_ms`).

## Notes

- **Delay comparison.** For each evaluated frame $t$ and delay $d$, the previous CoCs
  and the drive index are limited to frame $t - d$ (`contig`) or to frames $t - d, t - 2d, \dots$
  (`stride`, Fig. 10). The CoC of frame $t - d$ is added to the drive index inside the
  timed Decode call.
- **Real-time runs.** `cycles.jsonl` has one row per processed frame with the
  arrival, start, stage-end, and publication times (seconds from the run start),
  the stage times, the tokens, and the trajectory. The FlashDrive and RESPECT runs of Fig. 11 (Section V-E)
  were measured with the PyTorch 2.11 wheel setup; in that setup the `#4` run (`S4` in
  `realtime_flashdrive/`) gives a 2.38 s cycle and a 3.80 s event response latency,
  a separate measurement from the PyTorch 2.8 `#4` row in Fig. 11 (2.39 s and 3.81 s).
  Each configuration has 50 run conditions: 10 clips × 5 start positions, not
  50 independent clips. All paired intervals of
  Sections V-E and V-F resample clips (10,000 times, seed 260930); frames are matched by
  (clip, start position, frame).
- **`#0′`.** FlashDrive runs only with PyTorch 2.11, so `#0′` and a second `#4` run
  were measured together in that environment (`flashdrive_unified/`). Each clip is
  a new stream filled with the window of the frame before its first scored
  frame; Decode is timed from the end of FlashDrive's Prefill to the end of its
  generation. Each clip is processed twice, with the clip order reversed in the
  second pass; per-frame Decode time is the mean of the two passes. The `#4`
  comparison in that environment also averages two timed runs per frame. Memory is the increase over the allocation at the
  start of Decode, for every configuration.
- **Development runs.** During selection the datastore is one index of the
  clip's earlier CoCs and, for the scope `all`, the other development clips' CoCs,
  without timed insertion. In the paper's selection run the node-budget-128
  variants used a 192-slot continuation buffer; one frame whose CoC exceeds 64 tokens
  was measured with 256 slots, which this code uses for every budget-128 variant.
- **`supp_tile_boundary`** was measured with the kernels of the paper's measurements; an
  earlier check of the released kernels on one frame showed the same step with
  2–4 ms shorter times.
- **Input reconstruction.** In the two-frame example check, rebuilding inputs with
  the pinned dataset revision differed from the archived `.pt` files only in
  `image_frames` (maximum uint8 pixel difference 3; other fields equal). The cause
  was not established. Output-preservation checks compare `#4` with `K` on identical
  loaded inputs; they do not establish byte-identical reconstruction of paper inputs.
- **Timing** varies slightly between sessions; token/KV equality is checked against
  `K` within the same environment and on identical inputs.
- **Section V-G interval.** The interval of the `#0′ − #4` minADE6 difference resamples
  the clip means 10,000 times (one generator, seed 261009, event-window set first;
  `summarize_additions.py`). `#0`, `#4`, and `#0′` each generate six trajectories per frame
  from six different noise samples, and the six samples are the same on every frame.
- **Tests.** `PYTHONPATH=. python -m pytest tests` runs the trie, real-time metric, and
  bitwise kernel checks (16 tests; all pass in the tested Thor environment).
