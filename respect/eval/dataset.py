"""Evaluation and development frames of the paper and the per-frame candidate sources.

``data/{event,general}.jsonl`` list every frame of the evaluated clips and
``data/dev_{event,general}.jsonl`` every frame of the development clips, in
recording order: the scored frames and the frames before them that only supply
previous CoCs. ``reference_coc`` is the CoC of the default sequential run
(Alpamayo 1.5 public implementation, BF16, greedy), which serves as the
previous-frame CoC history and as the datastore content.

Candidate sources of a scored frame t whose newest usable CoC is from frame t-d:
  previous CoCs  ``contig``: the CoCs of the frames up to t-d in the same clip;
                 ``stride``: the CoCs of frames t-d, t-2d, ... (frames a single
                 GPU has finished when every frame takes d frame intervals).
                 d = 1 is the setting of the main evaluation.
  datastore      static index: every CoC of the development set, in file order
                 (event file, then general file); drive index: the same previous
                 CoCs of the clip. The CoC of frame t-d is added to the drive index
                 inside the timed Decode call.
The development set is never evaluated; the evaluated frames never enter the
static index.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

HISTORY_WINDOW = {"event": 100, "general": None, "dev": None}


@dataclass
class Frame:
    clip_id: str
    frame_idx: int
    t0_us: int
    scored: bool
    max_new_tokens: int
    reference_coc: list[int]
    meta: dict


def load_frames(path: Path) -> "OrderedDict[str, OrderedDict[int, Frame]]":
    clips: OrderedDict[str, OrderedDict[int, Frame]] = OrderedDict()
    for line in Path(path).read_text().splitlines():
        x = json.loads(line)
        f = Frame(x["clip_id"], int(x["frame_idx"]), int(x["t0_us"]), bool(x["scored"]),
                  int(x["max_new_tokens"]), [int(t) for t in x["reference_coc"]],
                  {k: v for k, v in x.items() if k not in ("clip_id", "frame_idx", "t0_us", "scored",
                                                           "max_new_tokens", "reference_coc")})
        clips.setdefault(f.clip_id, OrderedDict())[f.frame_idx] = f
    return clips


def input_path(input_dir: Path, clip_id: str, frame_idx: int) -> Path:
    return Path(input_dir) / f"{clip_id}_{frame_idx:04d}.pt"


def iterate(clips, set_name: str) -> Iterator[tuple[str, list[Frame], list[Frame]]]:
    """Yield (clip id, history frames before the first scored frame, scored frames)."""
    window = HISTORY_WINDOW["dev" if set_name.startswith("dev") else set_name]
    for cid in sorted(c for c in clips if any(f.scored for f in clips[c].values())):
        scored = [clips[cid][i] for i in sorted(i for i, f in clips[cid].items() if f.scored)]
        first = scored[0].frame_idx
        lo = first - window if window else -10 ** 9
        history = [clips[cid][i] for i in sorted(clips[cid]) if lo <= i < first]
        yield cid, history, scored


def static_sequences(data_dir: Path) -> list[tuple[int, ...]]:
    """Datastore static index content: all development CoCs in file order."""
    seqs = []
    for name in ("dev_event", "dev_general"):
        for line in (Path(data_dir) / f"{name}.jsonl").read_text().splitlines():
            if line.strip():
                seqs.append(tuple(int(t) for t in json.loads(line)["reference_coc"]))
    return seqs


@dataclass
class View:
    candidates: list[tuple[int, ...]]   # initial candidates, each ending with EOS
    drive_base: list[tuple[int, ...]]   # drive index before the newest CoC is added
    newest: tuple[int, ...] | None      # CoC of frame t-d, added inside the timed call
    n_entries: int


def view(clips, cid: str, frame_idx: int, n: int, eos: int, d: int = 1, mode: str = "contig") -> View:
    """Candidate sources of frame ``frame_idx`` when the newest usable CoC is from frame t-d."""
    from respect.candidates.history import strip_eos, unique_recent
    avail = [(i, strip_eos(f.reference_coc, eos)) for i, f in sorted(clips[cid].items()) if i < frame_idx]
    if mode == "contig":
        ent = [(i, s) for i, s in avail if i <= frame_idx - d]
    else:
        ent = [(i, s) for i, s in avail if (frame_idx - i) % d == 0 and i <= frame_idx - d]
    cands = [tuple(list(s) + [eos]) for _, s in unique_recent(ent, n)]
    full = [tuple(list(s) + [eos]) for _, s in ent]
    newest = [s for i, s in ent if i == frame_idx - d]
    newest_full = tuple(list(newest[0]) + [eos]) if newest else None
    base = list(full)
    if newest_full is not None and newest_full in base:
        base.remove(newest_full)          # removes the first equal CoC, as in the paper's measurement
    return View(cands, base, newest_full, len(ent))
