# Paper Figures, Tables, and Equations

This guide maps every figure, supplementary table, and equation in the RESPECT manuscript to its corresponding image file, generator script, and result source in this repository.

## 1. Paper Figures (Figs. 1–14)

| Paper figure | Section | Topic | Repository file / source | Generator script |
|---|---|---|---|---|
| Fig. 1 | Sec. I | Pedestrian-crowd event: camera view and stopping positions | Stopping-margin data in [`fig12_stopping_margin.png`](../figures/fig12_stopping_margin.png) panel (b); `summary_additions.json: stopping_margin.bbbe775f` | `scripts/summarize_additions.py` |
| Fig. 2 | Sec. I | Autoregressive decoding versus RESPECT Decode (initial and continuation verification) | [`fig2_execution.png`](../figures/fig2_execution.png) | `scripts/paper_method_figures.py` |
| Fig. 3 | Sec. III-A | Match with previous CoC and concentration of Decode time | [`fig3_concentration.png`](../figures/fig3_concentration.png) | `scripts/make_figures.py` |
| Fig. 4 | Sec. III-C | Recorded real-time processing timeline (`#0` run vs. `#4` run) | [`fig4_realtime_timeline.png`](../figures/fig4_realtime_timeline.png) | `scripts/make_figures.py` |
| Fig. 5 | Sec. IV-A | RESPECT Decode execution in one frame (initial trie and joint continuation trie) | [`fig5_method.png`](../figures/fig5_method.png) | `scripts/paper_method_figures.py` |
| Fig. 6 | Sec. IV-D | Physical KV slots and logical-order accumulation | [`fig6_kernel_order.png`](../figures/fig6_kernel_order.png) | `scripts/paper_method_figures.py` |
| Fig. 7 | Sec. V-B | Decode time of all configurations and target runs per frame | [`fig7_configurations.png`](../figures/fig7_configurations.png) | `scripts/make_figures.py` |
| Fig. 8 | Sec. V-B | Continuation candidate sources (mean and observed maximum, including per-frame oracle) | [`fig8_sources.png`](../figures/fig8_sources.png) | `scripts/make_figures.py` |
| Fig. 9 | Sec. V-C | Decode time in five-frame windows around event onset | [`fig9_event_timeline.png`](../figures/fig9_event_timeline.png) | `scripts/make_figures.py` |
| Fig. 10 | Sec. V-D | Decode-time reduction under execution delay $d$ | [`fig10_delay.png`](../figures/fig10_delay.png) | `scripts/make_figures.py` |
| Fig. 11 | Sec. V-E | Real-time stage times per cycle and event response latency | [`fig11_realtime_breakdown.png`](../figures/fig11_realtime_breakdown.png) | `scripts/paper_system_figures.py` |
| Fig. 12 | Sec. V-F | Computed stopping margins in four RSS-critical events | [`fig12_stopping_margin.png`](../figures/fig12_stopping_margin.png) | `scripts/make_figures.py` |
| Fig. 13 | Sec. V-G | Six trajectories per configuration on the pedestrian-crowd event frame (BEV rendering) | [`fig13_trajectories.png`](../figures/fig13_trajectories.png) | `scripts/paper_system_figures.py` |
| Fig. 14 | Sec. V-G | Frame-by-frame minADE6 against the public implementation (`#0`) | [`fig14_trajectory_error.png`](../figures/fig14_trajectory_error.png) | `scripts/make_figures.py` |

### Supplementary Figures

| Supplementary figure | Referenced in | Topic | Repository file | Generator script |
|---|---|---|---|---|
| `supp_n_select` | Sec. IV-B, [Supplement §2](supplement.md#2-setting-selection-on-the-development-set) | Development-set sweep over the number of previous frames $N$ | [`supp_n_select.png`](../figures/supp_n_select.png) | `scripts/make_figures.py` |
| `supp_tile_boundary` | Sec. IV-C, [Supplement §2](supplement.md#2-setting-selection-on-the-development-set) | Forward-pass time across the 64-to-65 query-tile boundary | [`supp_tile_boundary.png`](../figures/supp_tile_boundary.png) | `scripts/make_figures.py` |
| `supp_kernel_cost` | Sec. V-B, [Supplement §3](supplement.md#3-ablations) | Per-call duration with default PyTorch versus deterministic kernels | [`supp_kernel_cost.png`](../figures/supp_kernel_cost.png) | `scripts/make_figures.py` |

*Notes on figures:*
- Figures 2, 5, and 6 simplify model tokens to words for visual clarity; in Fig. 5, sentences A and B and the target CoC are recorded consecutive-frame examples, while sentence C and the FD-DFlash prediction tree illustrate branching.
- Figure 13 (`fig13_trajectories.png`) renders the top-down bird's-eye-view (BEV) trajectory comparison from [`results/paper/frame_trajectories/bbbe775f_0160.json`](../results/paper/frame_trajectories/bbbe775f_0160.json); the front-camera image panel requires gated dataset access and is not redistributed ([Supplement §7](supplement.md#7-trajectories-of-the-three-systems-on-the-same-frame)).

## 2. Result Tables and Supplementary Comparisons

| Topic | Paper section | Location in repository |
|---|---|---|
| Compared configurations (`#0` through `#5` and source ablations) | Sec. V-A | [Supplement §1](supplement.md#1-configurations) |
| Development-set continuation setting selection (tree size, top-$k$, node budget, datastore scope and key length) | Sec. IV-C | [Supplement §2](supplement.md#2-setting-selection-on-the-development-set); `summary.json: dev.one_at_a_time` |
| Offline Decode time and delay $d$ summary tables | Sec. V-B, V-D | [README Detailed results](../README.md#detailed-results); `summary.json: sets.*.configs`, `summary_additions.json: observed_delay` |
| Complete CoC agreement with each specified reference | Sec. V-B | [Supplement §4](supplement.md#4-output-preservation) |
| Real-time replay (`#0` run vs. `#4` run) | Sec. V-E | [README Detailed results](../README.md#detailed-results); `summary.json: section6.realtime.configs` |
| Real-time Prefill/Flow stage comparison (`#4`, FlashDrive BF16, FlashDrive + `#4`, `#5` RESPECT) | Sec. V-E | [README Detailed results](../README.md#detailed-results); `summary.json: section6.stage_link.configs` |
| Decode-region energy and peak CUDA memory | Sec. V-G | [Supplement §5](supplement.md#5-energy-and-memory) |
| Comparison with related speculative decoding and VLA acceleration methods | Sec. II | [Supplement §6](supplement.md#6-comparison-with-related-acceleration-methods) |

See [Paper item → source](../REPRODUCE.md#paper-item--source) in `REPRODUCE.md` for the exact script, JSON field, and numerical value backing every claim in the paper.

## 3. Equations

| Equation | Paper section | Topic | Documentation link |
|---|---|---|---|
| Eq. (1) | Sec. V-F | RSS longitudinal minimum safe distance $d_{\min}(\rho)$ | [Evaluation protocol §7](evaluation_protocol.md#7-rss-based-evaluation) |
| Decode cost decomposition | Sec. IV-A / IV-C | Initial verification plus continuation candidate-building and verification cost $T_t$ | [Method details §6](method_details.md#6-decode-cost) |

## 4. Regenerating All Figures

Run the [documented commands](../REPRODUCE.md#recompute-the-saved-results) to recompute `summary.json`, `summary_additions.json`, and all figures in `figures/` from the saved raw measurements without a GPU.
