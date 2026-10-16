# References

The references below list the primary models, datasets, speculative-decoding methods, numerical-execution documentation, and Responsibility-Sensitive Safety (RSS) sources cited across the paper and repository documentation.

| Source group | Used in |
|---|---|
| [Models and data](#models-and-data) | [Current-frame context](method_details.md#1-context-and-output-reference), [data selection](../data/README.md#selection-and-frame-scope), [FlashDrive comparison](supplement.md#8-flashdrive-stage-comparison-and-output-differences) |
| [Speculative decoding and retrieval](#speculative-decoding-and-retrieval) | [Acceptance](method_details.md#1-context-and-output-reference), [continuation candidates](method_details.md#4-continuation-candidate-sources) |
| [Numerical execution and memory](#numerical-execution-and-memory) | [Deterministic verification](method_details.md#7-deterministic-verification) |
| [RSS](#rss) | [RSS-based evaluation](evaluation_protocol.md#7-rss-based-evaluation) |

## Models and data

- NVIDIA. [Alpamayo 1.5](https://github.com/NVlabs/alpamayo1.5). Source repository, 2026.
- NVIDIA. [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles). Dataset and OOD reasoning annotations, 2026.
- z-lab. [Alpamayo-1.5-10B-DFlash](https://huggingface.co/z-lab/Alpamayo-1.5-10B-DFlash). Draft-model repository, 2026.
- Z. Li et al. [FlashDrive: Flash Vision-Language-Action Inference for Autonomous Driving](https://arxiv.org/abs/2608.12932). arXiv:2608.12932, 2026.

## Speculative decoding and retrieval

- Y. Leviathan, M. Kalman, and Y. Matias. [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html). *Proc. ICML*, pp. 19274–19286, 2023.
- Z. He, Z. Zhong, T. Cai, J. Lee, and D. He. [REST: Retrieval-Based Speculative Decoding](https://aclanthology.org/2024.naacl-long.88/). *Proc. NAACL*, pp. 1582–1595, 2024.

## Numerical execution and memory

- H. He and Thinking Machines Lab. [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). 2025.
- Z. Zhang et al. [Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch](https://arxiv.org/abs/2511.17826). arXiv:2511.17826, 2025.
- PyTorch. [Numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html). Documentation.
- NVIDIA. [Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html). CUDA documentation.
- P. Tillet, H. T. Kung, and D. Cox. [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://dl.acm.org/doi/10.1145/3315508.3329973). *Proc. MAPL*, pp. 10–19, 2019.
- W. Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). *Proc. SOSP*, pp. 611–626, 2023.

## RSS

- S. Shalev-Shwartz, S. Shammah, and A. Shashua. [On a Formal Model of Safe and Scalable Self-driving Cars](https://arxiv.org/abs/1708.06374). arXiv:1708.06374, 2017.
- I. Hasuo. [Responsibility-Sensitive Safety: An Introduction with an Eye to Logical Foundations and Formalization](https://arxiv.org/abs/2206.03418). arXiv:2206.03418, 2022.
- X. Xu, X. Wang, and X. Wu. “Calibration and Evaluation of Responsibility-Sensitive Safety Model on Autonomous Car-Following Maneuvers Using Naturalistic Driving Study Data.” *TRB Annual Meeting*, 2020.

The RESPECT implementation and result inventory are linked from the [repository README](../README.md) and [reproduction guide](../REPRODUCE.md).
