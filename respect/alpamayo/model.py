"""Alpamayo 1.5 loading, input construction and Prefill."""
from __future__ import annotations

import contextlib
import copy
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import LogitsProcessorList


@dataclass
class Target:
    model: Any
    processor: Any
    helper: Any
    eos: int      # end of the CoC: the token that opens the trajectory segment
    offset: int   # first trajectory-token id, excluded from CoC tokens
    size: int     # number of trajectory tokens


def load_target(model_id: str = "nvidia/Alpamayo-1.5-10B", local_files_only: bool = False) -> Target:
    """Load Alpamayo 1.5 in BF16 on the current CUDA device."""
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    import transformers.models.qwen3_vl.modeling_qwen3_vl as q3

    model = Alpamayo1_5.from_pretrained(model_id, dtype=torch.bfloat16,
                                        local_files_only=local_files_only).cuda().eval()
    processor = helper.get_processor(model.tokenizer)
    # Skip Qwen3-VL's data-dependent host checks inside forward. They read GPU
    # values on the CPU and do not change any computed value.
    q3.is_torchdynamo_compiling = lambda: True
    eos = int(model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>"))
    return Target(model, processor, helper, eos, model.config.traj_token_start_idx,
                  model.config.traj_vocab_size)


def make_raw_inputs(target: Target, inp: dict[str, Any]) -> dict[str, Any]:
    """Chat-template tokenization of one frame (images and ego history)."""
    frames_flat = inp["image_frames"].flatten(0, 1)
    messages = target.helper.create_message(frames=frames_flat, camera_indices=inp["camera_indices"])
    tokenized = target.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
        return_dict=True, return_tensors="pt")
    raw = {"tokenized_data": tokenized,
           "ego_history_xyz": inp["ego_history_xyz"],
           "ego_history_rot": inp["ego_history_rot"]}
    return target.helper.to_device(raw, "cuda")


def make_inputs(target: Target, inp: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Model inputs with the ego-history trajectory tokens fused into the prompt.

    Returns (prepared inputs for the VLM, raw inputs kept for the Flow module).
    """
    raw = make_raw_inputs(target, inp)
    data_copy = copy.deepcopy(raw)
    tokenized = data_copy["tokenized_data"]
    input_ids_raw = tokenized.pop("input_ids")
    input_ids = target.model.fuse_traj_tokens(
        input_ids_raw, {"ego_history_xyz": data_copy["ego_history_xyz"],
                        "ego_history_rot": data_copy["ego_history_rot"]})
    prepared = {"input_ids": input_ids,
                "attention_mask": tokenized.get("attention_mask"),
                "pixel_values": tokenized.get("pixel_values"),
                "image_grid_thw": tokenized.get("image_grid_thw")}
    return prepared, raw


def prefill(target: Target, prepared: dict[str, Any], autocast: bool = True, hidden_states: bool = True,
            seed_layers: tuple[list[int], int] | None = None) -> Any:
    """Vision Encoding and Prefill through the model's own ``generate`` for one step.

    The returned object carries the Prefill KV cache (``past_key_values``), the
    first CoC token (``sequences``) and the hidden states that seed FD-DFlash.
    With BF16 weights ``autocast=False`` gave a bitwise-identical KV cache on the tested frames;
    the fused Prefill kernels (``respect.system.fused``) run without autocast.
    ``hidden_states=False`` does not keep the hidden states (decoding without FD-DFlash).
    ``seed_layers=(layer_ids, n)`` keeps, instead of every hidden state, only the outputs of the given
    decoder layers at the last ``n`` positions, which is all that seeds FD-DFlash; they are returned in
    the same place (``hidden_states[0][i + 1]`` for layer ``i``).
    """
    from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor

    model = target.model
    config = copy.deepcopy(model.vlm.generation_config)
    config.do_sample = False
    config.top_p = 1.0
    config.temperature = 1.0
    config.num_return_sequences = 1
    config.max_new_tokens = 1
    config.return_dict_in_generate = True
    config.output_hidden_states = hidden_states
    config.output_scores = True
    config.output_logits = True
    config.pad_token_id = model.tokenizer.pad_token_id
    processors = LogitsProcessorList([ExpertLogitsProcessor(
        traj_token_offset=model.config.traj_token_start_idx,
        traj_vocab_size=model.config.traj_vocab_size)])
    kwargs = dict(input_ids=prepared["input_ids"], attention_mask=prepared.get("attention_mask"),
                  pixel_values=prepared.get("pixel_values"), image_grid_thw=prepared.get("image_grid_thw"),
                  generation_config=config, logits_processor=processors)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    if seed_layers is not None:
        config.output_hidden_states = False
        ids, n = seed_layers
        lm = model.vlm.model.language_model if hasattr(model.vlm.model, "language_model") else model.vlm.model
        kept: dict[int, torch.Tensor] = {}

        def keep(i):
            def hook(_m, _a, out):
                if i not in kept:                       # the Prefill forward only
                    h = out[0] if isinstance(out, tuple) else out
                    kept[i] = h[:, -n:, :].clone()
            return hook
        handles = [lm.layers[i].register_forward_hook(keep(i)) for i in ids]
    try:
        with torch.no_grad(), amp:
            out = model.vlm.generate(**{k: v for k, v in kwargs.items() if v is not None})
    finally:
        if seed_layers is not None:
            for h in handles:
                h.remove()
    if seed_layers is not None:
        step = [None] * (max(ids) + 2)
        for i in ids:
            step[i + 1] = kept[i]
        out.hidden_states = (tuple(step),)
    return out


def first_token(prefill_out: Any, prompt_len: int) -> int:
    """The first CoC token, fixed by the Prefill logits before Decode starts."""
    return int(prefill_out.sequences[0, prompt_len].item())


def prompt_length(prefill_out: Any) -> int:
    return int(prefill_out.past_key_values.get_seq_length())
