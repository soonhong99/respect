# Evaluation frames

`event.jsonl` (10 event-window clips, 1,400 frames, 400 scored) and `general.jsonl` (60 nominal-driving clips, 2,400 frames, 600 scored) list scored frames and preceding history from the evaluation clips.

`dev_event.jsonl` (15 event-window clips, 2,100 frames, 600 scored) and `dev_general.jsonl` (30 nominal-driving clips, 1,200 frames, 300 scored) list the corresponding frames from the disjoint development clips.

*(Note: in file and JSON key names, the nominal-driving set is denoted `general`.)*

All four files list frames of [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) in recording order.

Frames that are not scored only supply previous CoCs.

| Field | Meaning |
|---|---|
| `clip_id`, `frame_idx`, `t0_us` | frame identity; `t0_us` is the timestamp passed to the dataset loader |
| `scored` | frame included in the reported results |
| `max_new_tokens` | CoC token limit (128) |
| `reference_coc` | CoC token IDs of the baseline autoregressive run (`#0`, Alpamayo 1.5, BF16, greedy), ending with EOS |
| `reference_coc_text` | the same CoC decoded to text (special tokens skipped), used to compare FlashDrive's text output |
| `reference_minADE6_m`, `reference_minFDE6_m` | trajectory error of that run, Flow seeds 0–5 |
| `event_start_frame`, `frames_from_event`, `event_type` | event-window set only: the OOD-annotated start frame and the frame's offset from it |
| `start_frame` | development nominal-driving clips (`dev_general.jsonl`) only: the first scored frame |

The development files carry `clip_id`, `frame_idx`, `t0_us`, `scored`, `max_new_tokens`, `reference_coc`, and the event or start fields; trajectory errors and texts are not recorded for them.

The development clips never overlap the evaluated clips; `dev_selection.json` records how they were chosen (`scripts/collect_dev.py`). The datastore's static index holds every development CoC in file order (`dev_event.jsonl`, then `dev_general.jsonl`), and ties in continuation frequency keep the first-inserted continuation, so the file order matters.

## Selection and frame scope

The selection protocol below corresponds to Sections III-A and V-A of the paper.

Each clip is a continuous 20 s driving record with frames 0.1 s apart.

### Event-window evaluation

The OOD annotation supplies the event-start frame $E$; the 40 scored frames of each clip span $E - 20$ through $E + 19$.

Because OOD annotations are provided for training and validation data, evaluation uses validation clips and excludes the training data used for Alpamayo 1.5.

Of the 290 validation clips with an OOD annotation, 43 have both the 40 scored frames and 100 preceding frames.

These eligible clips comprise 25 construction/temporary-traffic-control, 7 unusual-vehicle, 6 pedestrian-crowd/proximity, 3 complex-intersection, and 2 cyclist/micromobility clips.

To prevent the largest category from dominating, evaluation selects two clips per type, matching the smallest category: 10 clips and 400 scored frames.

### Nominal-driving evaluation and development sets

Nominal-driving evaluation randomly selects 60 of 200 test clips and scores 10 consecutive frames per clip, giving 600 frames independently of event timing.

The event-window development set selects 15 clips with a fixed seed from the 33 eligible clips excluded from evaluation, capped at five per type: 5 construction, 5 unusual-vehicle, 4 pedestrian, and 1 intersection clip.

The same 40-frame rule gives 600 scored development frames.

The nominal-driving development set uses the next 30 clips in the same seeded random order after the evaluation selection, giving 300 scored frames.

### Scored frames, history, and adjacent pairs

| File | Clips | Scored frames per clip | Preceding frames per clip | Total listed frames |
|---|---:|---:|---:|---:|
| `event.jsonl` | 10 | 40 | 100 | 1,400 |
| `general.jsonl` | 60 | 10 | 30 | 2,400 |
| `dev_event.jsonl` | 15 | 40 | 100 | 2,100 |
| `dev_general.jsonl` | 30 | 10 | 30 | 1,200 |

The CoC-reuse analysis (Section III-A, Fig. 3(a)) compares consecutive baseline CoCs across each evaluation clip's scored frames and preceding history: $10 \times (140 - 1) = 1,390$ event-window pairs and $60 \times (40 - 1) = 2,340$ nominal-driving pairs.

These are adjacent-pair counts for reuse analysis, not additional frames in the 400/600-frame performance evaluation.

The development files supply 3,300 CoCs in total for the static datastore, including preceding frames; the scored development scope is 900 frames.

[Evaluation protocol](../docs/evaluation_protocol.md) explains offline candidate history, development-time aggregation, timing boundaries, real-time execution, and trajectory/RSS evaluation.

## Input availability and delay history

Sensor data and ground-truth trajectories are not redistributed; inputs are rebuilt from the dataset with the listed `clip_id` and `t0_us`.

`general_history_d40.jsonl`: 1,140 earlier frames of the nominal-driving (`general.jsonl`) clips (frames $t - 40, t - 80, \dots$ down to frame 18, the first frame the dataset loader accepts), decoded with the baseline autoregressive run (`#0`) of the public implementation by `scripts/collect_history.py`; used only for the nominal-driving set's delay comparison at $d = 40$.

`general_history_d16_d23.jsonl`: 2,327 further earlier frames of the nominal-driving (`general.jsonl`) clips (frames $t - 16, t - 32, \dots, t - 20, t - 40, \dots, t - 23, t - 46, \dots$ down to frame 18 that `general_history_d40.jsonl` does not already hold), collected the same way; used with `general_history_d40.jsonl` for the nominal-driving set's delay comparison at $d = 16, 20, 23$ (`results/paper/delay_observed/`).
