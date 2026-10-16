"""RESPECT settings used for every result in the paper.

N, the FD-DFlash tree size and top-k, the node budget and the datastore key
length were chosen on the development set (scripts/select_hparams.py), not on
the evaluated frames.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RespectConfig:
    num_prev_frames: int = 4          # N: previous frames whose CoCs form the initial trie
    node_budget: int = 64             # B: nodes per follow-up trie, one query tile of the attention kernel
    draft_tree_nodes: int = 16        # FD-DFlash tree size, excluding the root
    draft_topk: int = 4               # tokens considered per predicted position
    datastore_key_len: int = 2        # longest datastore key; shortened to 1 on a miss
    follow_capacity: int = 192        # slots after the prompt used by follow-up verification
    buffer_capacities: tuple = (192, 256, 512, 1024)  # window sizes of the fixed KV buffer
    max_new_tokens: int = 128         # Alpamayo 1.5 default
