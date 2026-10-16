"""Output-preserving system optimizations outside Decode (Section VI-D).

These change only how Prefill and Flow are executed, not what they compute: the
Prefill KV cache and the trajectory stay bitwise identical to the public
implementation.  They are independent of RESPECT.
"""
