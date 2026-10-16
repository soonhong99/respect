"""RESPECT: REuse-based SPECulative decoding with Temporal drafts."""
import importlib

__version__ = "0.1.0"

_EXPORTS = {
    "CocHistory": "respect.candidates.history",
    "Datastore": "respect.candidates.datastore",
    "RespectConfig": "respect.config",
    "FixedKVBuffer": "respect.decoding.kv_buffer",
    "DecodeResult": "respect.decoding.respect",
    "FrameState": "respect.decoding.respect",
    "RespectDecoder": "respect.decoding.respect",
    "clone_cache": "respect.decoding.sequential",
    "sequential": "respect.decoding.sequential",
    "dedicated_kernels": "respect.kernels.patch",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    # Lazy imports keep the CPU-only parts (trie, datastore) usable without torch/transformers.
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(name)
