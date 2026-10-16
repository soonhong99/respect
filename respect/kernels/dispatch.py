"""Route the target's SDPA attention to the trie-verification kernel.

The key axis is split at the Prompt length PL: slots before PL are the prompt
segment, slots from PL on are the node segment. The node segment is always
padded to a multiple of NSTEP and the query rows to a multiple of MSTEP, so the
reduction sizes do not depend on how many nodes a call carries. A 4-D mask from
the caller is used as is; without one, a causal mask is built.

Calls that do not fit (batch > 1, the draft model's own attention, sizes beyond
the fixed limits) fall back to the original SDPA function.
"""
import torch
import transformers.integrations.sdpa_attention as SA

from respect.kernels.attention import fused_attn_hyb

ORIG = SA.sdpa_attention_forward
BM = 64
MSTEP = 64
MMAX = 1024
NSTEP = 32
NMAX = 4096
BNB = 16
STATE = {"PL": None}
CFG = dict(BM=BM, BN=128, num_warps=4, num_stages=1)
STATS = {"kernel": 0, "fallback": 0}
_BUF = {}
_PROMPT_OK = {}


def _fb():
    STATS["fallback"] += 1


def _buf(name, shape, dtype, device):
    # One buffer per shape, so that a shape change never reallocates a buffer in use.
    key = (name, tuple(shape), dtype, device)
    b = _BUF.get(key)
    if b is None:
        b = torch.zeros(shape, dtype=dtype, device=device)
        _BUF[key] = b
    return b


def _allow_from(am):
    a = am[0, 0] if am.dim() == 4 else am
    return a if a.dtype == torch.bool else (a == 0)


def _prompt_all_open(allow, PL, key):
    v = _PROMPT_OK.get(key)
    if v is None:
        v = bool(allow[:, :PL].all())
        _PROMPT_OK[key] = v
        if len(_PROMPT_OK) > 8:
            _PROMPT_OK.clear()
            _PROMPT_OK[key] = v
    return v


def unified_forward(module, query, key, value, attention_mask,
                    dropout=0.0, scaling=None, is_causal=None, **kwargs):
    PL = STATE["PL"]
    if PL is None or query.shape[0] != 1 or dropout != 0.0 or kwargs.get("output_attentions", False):
        _fb()
        return ORIG(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)
    q_len = query.shape[2]
    n_kv = key.shape[2]
    n_node = n_kv - PL
    if n_node < q_len or n_node < 0:
        _fb()
        return ORIG(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)
    if q_len > MMAX or n_node > NMAX:
        _fb()
        return ORIG(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)
    dev = query.device
    H = query.shape[1]
    D = query.shape[3]
    HK = key.shape[1]

    node_allow = None
    if attention_mask is not None and attention_mask.dim() == 4 \
            and attention_mask.shape[-1] == n_kv and attention_mask.shape[-2] == q_len:
        allow = _allow_from(attention_mask)
        if not _prompt_all_open(allow, PL, (attention_mask.data_ptr(), tuple(attention_mask.shape))):
            _fb()
            return ORIG(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)
        node_allow = allow[:, PL:]
    elif attention_mask is not None and attention_mask.dim() == 4:
        _fb()
        return ORIG(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)

    scale = scaling if scaling is not None else D ** -0.5
    MP = ((q_len + MSTEP - 1) // MSTEP) * MSTEP
    q = _buf("q", (H, MP, D), query.dtype, dev)
    q.zero_()
    q[:, :q_len, :] = query[0]
    NN = max(NSTEP, ((max(n_node, 1) + NSTEP - 1) // NSTEP) * NSTEP)
    kn = _buf("kn", (HK, NN, D), key.dtype, dev)
    kn.zero_()
    vn = _buf("vn", (HK, NN, D), value.dtype, dev)
    vn.zero_()
    if n_node > 0:
        kn[:, :n_node, :] = key[0][:, PL:, :]
        vn[:, :n_node, :] = value[0][:, PL:, :]
    msk = _buf("msk", (MP, NN), torch.bool, dev)
    msk.zero_()
    if node_allow is not None:
        msk[:q_len, :n_node] = node_allow
    else:
        first = n_node - q_len
        rows = torch.arange(q_len, device=dev)[:, None] + first
        cols = torch.arange(NN, device=dev)[None, :]
        msk[:q_len, :] = cols <= rows
    msk[q_len:, 0] = True
    try:
        kpv = key[0][:, :PL, :]
        vpv = value[0][:, :PL, :]
        out = fused_attn_hyb(q, kpv, vpv, kn, vn, msk, scale, BNB=BNB, **CFG)
    except Exception:
        _fb()
        return ORIG(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)
    STATS["kernel"] += 1
    return out[:, :q_len, :][None].transpose(1, 2).contiguous(), None


def install(PL):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    STATE["PL"] = PL
    _PROMPT_OK.clear()
    SA.sdpa_attention_forward = unified_forward
    ALL_ATTENTION_FUNCTIONS["sdpa"] = unified_forward


def uninstall():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    STATE["PL"] = None
    _PROMPT_OK.clear()
    SA.sdpa_attention_forward = ORIG
    ALL_ATTENTION_FUNCTIONS["sdpa"] = ORIG
