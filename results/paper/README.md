# Raw results of the paper

| Path | Content |
|---|---|
| `main/frames_{event,general}.jsonl` | evaluation set, one row per scored frame: the ten configurations (both run orders), target runs, memory, power, energy, KV and token check of `#4` against `deterministic-kernel autoregressive decoding` (`K`), Flow minADE6 / minFDE6 and Flow time of `#4`, the delay comparison (`lag`), Prefill and Vision Encoding time, and the public inference function (`official`) |
| `dev/frames_{event,general}.jsonl` | development set, one row per scored frame: `#4` at every $N$ with the continuation settings at the chosen values (both run orders), with token and KV checks |
| `dev/followup_{event,general}.jsonl`, `dev/selection.json` | every continuation setting at the chosen $N$ on the development frames that need continuation verification at that $N$, and the values chosen by the pre-set rule |
| `dev/tile_boundary.json` | trie-verification forward time against node count on four development frames (Section IV-C, `supp_tile_boundary.png`) |
| `flashdrive_unified/fd.jsonl` | FlashDrive BF16 (`#0′`) on the evaluation frames (`baselines/flashdrive/offline_unified.py`), one row per frame and run: Decode time, verified blocks, tokens, Decode memory increase, power, and energy |
| `flashdrive_unified/respect.jsonl` | the same frames in the same process (PyTorch 2.11): `#4` runs and their check against `deterministic-kernel autoregressive decoding`, and the bit-exact Prefill/Flow check against the public implementation |
| `flashdrive_unified/fill_check.json` | the first windows of `#0′` and their overlap check against the evaluation inputs |
| `delay/frames_{event,general}.jsonl` | the delay comparison extended to $d = 20, 30, 40$ (Fig. 10), one row per scored frame; the nominal-driving set's (`general`) `40_stride` uses the earlier frames of `data/general_history_d40.jsonl` |
| `delay_observed/frames_{event,general}.jsonl` | the delay comparison at $d = 16, 20, 23$ (`stride`); $d = 16$ and $23$ are the median delays of the RESPECT and `#4` real-time runs, respectively (Section V-D, Fig. 10); the nominal-driving set (`general`) uses the earlier frames of `data/general_history_d40.jsonl` and `data/general_history_d16_d23.jsonl` |
| `flashdrive_quality/fd_quality.jsonl`, `fd_quality_summary.json` | FlashDrive BF16 (`#0′`) with six trajectories and the same Flow noise on every frame (`baselines/flashdrive/quality_fd.py --fixed-seed 0`, Section V-G): CoC tokens and minADE6 / minFDE6 per frame; trajectories removed |
| `dev_ablation/frames_{event,general}.jsonl` | every configuration on the development frames ([Supplement §3](../../docs/supplement.md#3-ablations)); two extra variants (`S4noft`: without the first-token condition, `S4noctx`: without the FD-DFlash context update) are not used in the paper |
| `realtime/cycles.jsonl`, `realtime/verify.jsonl` | real-time runs (Section III-C, Section V-E, Figs. 4 and 11): one row per processed frame, and the autoregressive check of every processed frame |
| `realtime/rss_leads.json` | ego speed and lead object at the event-start frame of every event clip (Section V-F) |
| `realtime_flashdrive/cycles.jsonl`, `verify.jsonl` | FlashDrive's stages linked with RESPECT Decode in real time (Section V-E, Fig. 11, `baselines/flashdrive/stage_link.py`): eight configurations, one row per processed frame, and the check of every processed frame |
| `realtime_fast/cycles.jsonl`, `verify.jsonl` | RESPECT (`#5`): the `#4` run with the bit-exact Prefill and Flow (Section V-E, Fig. 11, `scripts/run_realtime_fast.py`), and the token and trajectory-bit check against a recomputation with public Prefill/Flow and `deterministic-kernel autoregressive decoding` (`K`) |
| `memory/{S0,K,FD,S4,S5}.jsonl`, `*_summary.json` | peak memory of `#0`, `deterministic-kernel autoregressive decoding` (`K`), `#0′`, `#4`, and `#5` (RESPECT), each in its own process (`scripts/measure_memory.py`, `baselines/flashdrive/memory_fd.py`; Section V-G) |
| `frame_trajectories/bbbe775f_0160.json` | the frame of Fig. 13 (pedestrian-crowd event): six trajectories each of `#0`, RESPECT (`#4`), and `#0′` with the same Flow noise, and the recorded path |
| `summary.json` | output of `scripts/summarize.py` |
| `summary_additions.json` | output of `scripts/summarize_additions.py` (observed delays, minADE6 of `#0′`, $N$ under delay, signed stopping margins, Fig. 13 frame) |

Sampled trajectories are omitted to keep the repository small, except the frame of Fig. 13; `run_main.py --flow` regenerates them. Field names follow `scripts/run_main.py`; configuration codes are listed in [REPRODUCE.md](../../REPRODUCE.md).

See [Paper Figures, Tables, and Equations](../../docs/manuscript_map.md) for the complete index of figures and tables.

The earlier development pairwise-kernel run behind the `3/400` count in Section IV-D and the auxiliary Flow-weight swap check noted in [Supplement §8](../../docs/supplement.md#8-flashdrive-stage-comparison-and-output-differences) are documented in [Method details §8](../../docs/method_details.md#8-kernel-reduction-order-comparison) and [Supplement §8](../../docs/supplement.md#8-flashdrive-stage-comparison-and-output-differences).
