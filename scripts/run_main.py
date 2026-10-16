#!/usr/bin/env python3
"""Main evaluation of the paper: the ten configurations, the delay comparison and the
stage times of the public implementation on every scored frame, in one run.

Configurations (Table III and Section V-A of the paper)
  S0    #0   default sequential decoding                       (default kernels)
  S1    #1   initial verification, then sequential decoding   (default kernels)
  S2b   #2   #1 + FD-DFlash follow-up candidates              (default kernels)
  S2a   #2'  #1 + datastore follow-up candidates              (default kernels)
  S3    #3   #1 + FD-DFlash + datastore                       (default kernels)
  K          sequential decoding, output-preservation reference (dedicated kernels)
  S1fK       #4 w/o both sources: follow-up verification, one token per run
  S2bK       #4 w/o datastore
  S2aK       #4 w/o FD-DFlash
  S4    #4   RESPECT                                           (dedicated kernels)

Every frame runs all configurations twice, in forward and reverse order; the
frame value of a metric is the mean of the two runs. Decode time is the
wall-clock between CUDA synchronizations around the Decode call; it includes
adding the newest CoC to the datastore's drive index. Prefill, the copy of the
Prefill KV into the fixed buffer, the static datastore index (built once from the
development set before the run) and Flow are outside the timed region.

Delay comparison (Section VI): with the dedicated kernels, #4 is rerun with the
previous CoCs and the drive index limited to frame t-d for d = 2, 3, 5, 10, 20 in
two modes (contig: every frame up to t-d; stride: frames t-d, t-2d, ...).
With --delay-only --delays 20,30,40 only the delay comparison is run (Figure 12
points d = 30, 40); the reference tokens are the dedicated-kernel sequential run of the
same frame, computed outside timing.

Development set (Section V-C): --sets dev_event,dev_general with the development
inputs runs every configuration on the development frames.

With --flow, #4 and K are rerun outside timing to compare the committed KV
bitwise, Flow is run on #4's cache with seeds 0-5 (each seed timed), and the
stage times of the public implementation are recorded: Vision Encoding and
Prefill of this frame, and one call of the public inference function
(sample_trajectories_from_data_with_vlm_rollout with the arguments of its
example: temperature 0.6, top-p 0.98, bf16 autocast) with one trajectory on every
frame and six trajectories on every tenth scored frame of a set.

Output: <out>/frames_<set>.jsonl, one row per scored frame, and <out>/contract.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respect.alpamayo.flow import ground_truth, min_ade_fde, trajectory  # noqa: E402
from respect.alpamayo.model import first_token, load_target, make_inputs, prefill, prompt_length  # noqa: E402
from respect.alpamayo.power import PowerSampler  # noqa: E402
from respect.candidates.datastore import Datastore  # noqa: E402
from respect.candidates.fd_dflash import find_snapshot, load_fd_dflash  # noqa: E402
from respect.config import RespectConfig  # noqa: E402
from respect.decoding.kv_buffer import FixedKVBuffer  # noqa: E402
from respect.decoding.respect import FrameState, RespectDecoder  # noqa: E402
from respect.decoding.sequential import clone_cache, sequential  # noqa: E402
from respect.eval.dataset import input_path, iterate, load_frames, static_sequences, view  # noqa: E402
from respect.kernels import patch as kernels  # noqa: E402

STOCK = ["S0", "S1", "S2b", "S2a", "S3"]
KERNEL = ["K", "S1fK", "S2bK", "S2aK", "S4"]
SEEDS = [0, 1, 2, 3, 4, 5]
DS = [2, 3, 5, 10, 20]
MODES = ["contig", "stride"]
USE_DS = {"S2a", "S3", "S2aK", "S4"}
FWD = STOCK + ["install"] + KERNEL + ["uninstall"]
REV = ["install"] + KERNEL[::-1] + ["uninstall"] + STOCK[::-1]
# (datastore, FD-DFlash, follow-up mode) of each non-sequential configuration
VARIANT = {"S1": (False, False, "sequential"), "S2b": (False, True, "verify"), "S2a": (True, False, "verify"),
           "S3": (True, True, "verify"), "S1fK": (False, False, "verify"), "S2bK": (False, True, "verify"),
           "S2aK": (True, False, "verify"), "S4": (True, True, "verify")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--inputs", required=True, help="directory of <clip_id>_<frame:04d>.pt input files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="event,general")
    ap.add_argument("--flow", action="store_true")
    ap.add_argument("--limit-frames", type=int, default=0, help="stop each set after this many frames")
    ap.add_argument("--delays", default="2,3,5,10,20", help="delays d of the delay comparison")
    ap.add_argument("--delay-only", action="store_true", help="run only the delay comparison")
    ap.add_argument("--history", help="extra unscored rows of earlier frames (collect_history.py), merged into the clips")
    ap.add_argument("--modes", default="contig,stride", help="candidate modes of the delay comparison")
    ap.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    args = ap.parse_args()

    cfg = RespectConfig()
    DS[:] = [int(x) for x in args.delays.split(",")]
    MODES[:] = args.modes.split(",")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = load_target(args.model, local_files_only=True)
    model = target.model
    par = next(model.parameters())
    draft, mask_emb = load_fd_dflash(find_snapshot(), par.dtype, par.device)
    tc = time.perf_counter()
    static = Datastore(static_sequences(Path(args.data)), gmax=cfg.datastore_key_len)
    static_ms = (time.perf_counter() - tc) * 1000
    vis = VisionTimer(model)
    gen_config0 = copy.deepcopy(model.vlm.generation_config)
    (out_dir / "contract.json").write_text(json.dumps(dict(
        config=cfg.__dict__, stock=STOCK, kernel=KERNEL, seeds=SEEDS, delays=DS, modes=MODES,
        static_datastore=dict(build_ms=round(static_ms, 1), entries=static.num_entries()),
        vision_module=vis.name, torch=torch.__version__,
        device=torch.cuda.get_device_name(0), argv=sys.argv, started=time.strftime("%Y-%m-%d %H:%M:%S"),
        clocks=_sh("sudo -n /usr/bin/jetson_clocks --show")[-1500:], nvpmodel=_sh("nvpmodel -q")), indent=2))

    sampler = PowerSampler()
    sampler.start()
    time.sleep(0.5)
    buf: FixedKVBuffer | None = None
    dec: RespectDecoder | None = None
    t_start = time.perf_counter()
    nrow = 0

    def decode(name: str, fs: FrameState, v, pf) -> tuple[list[int], dict]:
        cands = v.candidates
        ds = static.with_drive(v.drive_base) if name in USE_DS else None      # outside timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if name in ("S0", "K"):
            tk, cache = sequential(model, clone_cache(pf.past_key_values), fs.first_token, fs.prompt_len,
                                   target.eos, target.offset, target.size, fs.max_tokens)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            del cache
            r = dict(target_runs=len(tk))
        else:
            use_ds, use_draft, follow = VARIANT[name]
            res = dec.decode(fs, cands, ds, use_draft=use_draft, follow=follow,
                             new_coc=v.newest if use_ds else None)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            tk = res.tokens
            r = dict(front_nodes=res.initial_nodes, front_cands=res.initial_candidates, accepted=res.accepted_initial,
                     tail_tokens=len(tk) - res.accepted_initial, rounds=res.follow_rounds,
                     target_runs=res.target_runs, front_ms=round(res.initial_ms, 3), tail_ms=round(res.follow_ms, 3),
                     draft_ms=res.draft_ms, lookup_ms=res.lookup_ms, verify_ms=res.verify_ms,
                     ds_insert_ms=res.datastore_insert_ms,
                     round_nodes=res.round_nodes, round_acc=res.round_accepted)
        ms = (t1 - t0) * 1000
        r.update(ms=round(ms, 3), n_tokens=len(tk), ms_per_token=round(ms / max(len(tk), 1), 3), _t=(t0, t1))
        return [int(x) for x in tk], r

    def run_cfg(name, fs, v, pf):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_allocated()
        tk, r = decode(name, fs, v, pf)
        t0, t1 = r.pop("_t")
        r.update(tokens=tk, peak_alloc_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                 peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
                 peak_delta_mib=round((torch.cuda.max_memory_allocated() - m0) / 2 ** 20, 1))
        pw = sampler.window(t0, t1) or {}
        if pw:
            r.update(power_module_w=pw["module_w"], power_gpu_w=pw["gpu_w"],
                     energy_j=round(pw["module_w"] * r["ms"] / 1000, 4))
        return r

    for set_name in args.sets.split(","):
        clips = load_frames(Path(args.data) / f"{set_name}.jsonl")
        if args.history:
            extra = load_frames(Path(args.history))
            for cid, frs in extra.items():
                if cid in clips:
                    merged = {**frs, **clips[cid]}
                    clips[cid].clear()
                    clips[cid].update(sorted(merged.items()))
        fpath = out_dir / f"frames_{set_name}.jsonl"
        done = set()
        if fpath.exists():
            for line in fpath.read_text().splitlines():
                x = json.loads(line)
                done.add((x["clip_id"], x["frame_idx"]))
        fp = fpath.open("a")
        nset = len(done)
        tot = sum(1 for c in clips.values() for f in c.values() if f.scored)
        k_set = 0
        for cid, hist_frames, scored in iterate(clips, set_name):
            for fr in scored:
                k_set += 1
                if args.limit_frames and k_set > args.limit_frames:
                    break
                if (cid, fr.frame_idx) in done:
                    continue
                inp = torch.load(input_path(Path(args.inputs), cid, fr.frame_idx), weights_only=False)
                prepared, raw = make_inputs(target, inp)
                views = {1: view(clips, cid, fr.frame_idx, cfg.num_prev_frames, target.eos)}
                for d in DS:
                    for m in MODES:
                        views[(d, m)] = view(clips, cid, fr.frame_idx, cfg.num_prev_frames, target.eos, d, m)
                v1 = views[1]
                assert not kernels.is_installed()
                assert model.vlm.generation_config.to_dict() == gen_config0.to_dict()
                vis.start()
                torch.cuda.synchronize()
                tp = time.perf_counter()
                pf = prefill(target, prepared)
                torch.cuda.synchronize()
                prefill_ms = (time.perf_counter() - tp) * 1000
                vision_ms = vis.stop()
                PL = prompt_length(pf)
                if buf is None:
                    buf = FixedKVBuffer(PL, cfg.buffer_capacities)
                    dec = RespectDecoder(target, buf, cfg, draft, mask_emb)
                assert buf.PL == PL, "the fixed buffer assumes one prompt length"
                buf.allocate(pf.past_key_values)
                torch.cuda.synchronize()
                tq = time.perf_counter()
                buf.write_prefill(pf.past_key_values)
                torch.cuda.synchronize()
                copy_ms = (time.perf_counter() - tq) * 1000
                fs = FrameState(prepared, pf, PL, first_token(pf, PL), fr.max_new_tokens)

                if args.delay_only:
                    kernels.install(model, PL)
                    try:
                        k, _c = sequential(model, clone_cache(pf.past_key_values), fs.first_token, PL, target.eos,
                                           target.offset, target.size, fs.max_tokens)
                        k = [int(x) for x in k]
                        del _c
                        # a delay with no previous frame at t-d in this clip is not measured
                        dkeys = [key for key in views if key != 1 and views[key].n_entries > 0]
                        if nrow == 0:
                            for key in dkeys:
                                decode("S4", fs, views[key], pf)
                        dres = {}
                        for pas, order in (("fwd", dkeys), ("rev", dkeys[::-1])):
                            for key in order:
                                tk, r = decode("S4", fs, views[key], pf)
                                r.pop("_t")
                                r["match_K"] = (tk == k)
                                for kk in ("round_nodes", "round_acc"):
                                    r.pop(kk, None)
                                dres.setdefault(key, {})[pas] = r
                    finally:
                        kernels.uninstall(model)
                    lag = {f"{d}_{m}": dict(n_entries=views[(d, m)].n_entries, n_cand=len(views[(d, m)].candidates),
                                            has_newest=views[(d, m)].newest is not None, **dres[(d, m)],
                                            ms_mean2=round((dres[(d, m)]["fwd"]["ms"] + dres[(d, m)]["rev"]["ms"]) / 2, 3))
                           for d, m in dkeys}
                    fp.write(json.dumps(dict(set=set_name, clip_id=cid, frame_idx=fr.frame_idx, prefill_len=PL,
                                             tokens_K=k, lag=lag)) + "\n")
                    fp.flush()
                    nrow += 1
                    nset += 1
                    bad = [kk for kk, x in lag.items() if not (x["fwd"]["match_K"] and x["rev"]["match_K"])]
                    print("[%s %d/%d] %s f%d  %s | %s  (%.1f min)" % (
                        set_name, nset, tot, cid[:8], fr.frame_idx,
                        "  ".join("%s %.0f" % (kk, x["ms_mean2"]) for kk, x in lag.items()),
                        "match K" if not bad else "MISMATCH " + ",".join(bad), (time.perf_counter() - t_start) / 60),
                          flush=True)
                    del inp, prepared, raw, pf
                    continue

                if nrow == 0:                                 # warm up every configuration once
                    for nm in STOCK:
                        run_cfg(nm, fs, v1, pf)
                    kernels.install(model, PL)
                    for nm in KERNEL:
                        run_cfg(nm, fs, v1, pf)
                    for key in views:
                        if key != 1:
                            decode("S4", fs, views[key], pf)
                    kernels.uninstall(model)
                kernels.install(model, PL)
                run_cfg("S4", fs, v1, pf)                    # per-frame warm-up
                kernels.uninstall(model)
                run_cfg("S3", fs, v1, pf)
                res = {}
                for pas, order in (("fwd", FWD), ("rev", REV)):
                    for step in order:
                        if step == "install":
                            kernels.install(model, PL)
                        elif step == "uninstall":
                            kernels.uninstall(model)
                        else:
                            assert (step in KERNEL) == kernels.is_installed()
                            res[(pas, step)] = run_cfg(step, fs, v1, pf)
                s0, k = res[("fwd", "S0")]["tokens"], res[("fwd", "K")]["tokens"]
                cfgs = {}
                for name in STOCK + KERNEL:
                    f_, r_ = res[("fwd", name)], res[("rev", name)]
                    tk, tk_r = f_.pop("tokens"), r_.pop("tokens")
                    cfgs[name] = dict(fwd=f_, rev=r_, n_tokens=len(tk), same_fwd_rev=(tk == tk_r),
                                      match_S0=(tk == s0), match_K=(tk == k), match_reference=(tk == fr.reference_coc),
                                      ms_mean2=round((f_["ms"] + r_["ms"]) / 2, 3),
                                      tokens=tk if (name in ("S0", "K", "S3", "S4") or tk != s0) else None)
                # delay comparison (dedicated kernels, both orders)
                dkeys = [key for key in views if key != 1]
                dres: dict = {}
                kernels.install(model, PL)
                try:
                    for pas, order in (("fwd", dkeys), ("rev", dkeys[::-1])):
                        for key in order:
                            tk, r = decode("S4", fs, views[key], pf)
                            r.pop("_t")
                            r["match_K"] = (tk == k)
                            for kk in ("round_nodes", "round_acc"):
                                r.pop(kk, None)
                            dres.setdefault(key, {})[pas] = r
                finally:
                    kernels.uninstall(model)
                lag = {f"{d}_{m}": dict(n_entries=views[(d, m)].n_entries, n_cand=len(views[(d, m)].candidates),
                                        has_newest=views[(d, m)].newest is not None, **dres[(d, m)],
                                        ms_mean2=round((dres[(d, m)]["fwd"]["ms"] + dres[(d, m)]["rev"]["ms"]) / 2, 3))
                       for d, m in dkeys}
                flow = official = None
                if args.flow:
                    flow = check_flow(model, target, dec, buf, fs, static.with_drive(v1.drive_base), v1, pf,
                                      prepared, raw, inp)
                    official = run_official(model, target, inp, vis, with6=(k_set % 10 == 1))
                    model.vlm.generation_config = copy.deepcopy(gen_config0)
                row = dict(set=set_name, clip_id=cid, frame_idx=fr.frame_idx, prefill_len=PL, cap=fr.max_new_tokens,
                           n_unique_prev=len(v1.candidates), prefill_ms=round(prefill_ms, 3),
                           vision_ms=[round(x, 3) for x in vision_ms],
                           prefill_copy_ms=round(copy_ms, 3), meta=fr.meta,
                           reference_minADE6_m=fr.meta.get("reference_minADE6_m"),
                           reference_minFDE6_m=fr.meta.get("reference_minFDE6_m"),
                           tokens_K=k, flow=flow, configs=cfgs, lag=lag, official=official)
                fp.write(json.dumps(row) + "\n")
                fp.flush()
                nrow += 1
                nset += 1
                bad = [nm for nm in KERNEL if not (cfgs[nm]["match_K"] and cfgs[nm]["same_fwd_rev"])]
                bad += [kk for kk, x in lag.items() if not (x["fwd"]["match_K"] and x["rev"]["match_K"])]
                print("[%s %d/%d] %s f%d  %s  d_max %.0f ms | kernel configs %s%s  (%.1f min)" % (
                    set_name, nset, tot, cid[:8], fr.frame_idx,
                    "  ".join("%s %.0f" % (nm, cfgs[nm]["ms_mean2"]) for nm in STOCK + KERNEL),
                    lag[f"{DS[-1]}_stride"]["ms_mean2"],
                    "match K" if not bad else "MISMATCH " + ",".join(bad),
                    "" if flow is None else "  KV %s  minADE6 %.3f" % (flow["kv_equal"], flow["s4_minADE6_m"]),
                    (time.perf_counter() - t_start) / 60), flush=True)
                del inp, prepared, raw, pf
        fp.close()
    sampler.stop()
    print("done. frames %d, %.1f min" % (nrow, (time.perf_counter() - t_start) / 60), flush=True)


def check_flow(model, target, dec, buf, fs, ds, v, pf, prepared, raw, inp) -> dict:
    """Outside timing: bitwise KV check of #4 against K, and Flow trajectories of #4 (each seed timed)."""
    gt = ground_truth(inp)
    kernels.install(model, fs.prompt_len)
    try:
        tk_k, cache_k = sequential(model, clone_cache(pf.past_key_values), fs.first_token, fs.prompt_len,
                                   target.eos, target.offset, target.size, fs.max_tokens)
        tk_k = [int(x) for x in tk_k]
        tk_4 = dec.decode(fs, v.candidates, ds, use_draft=True, new_coc=v.newest).tokens
        Lk, L4 = int(cache_k.get_seq_length()), fs.prompt_len + len(tk_4)
        L = min(Lk, L4)
        kv_eq = all(torch.equal(cache_k.layers[i].keys[..., :L, :], buf.base_k[i][..., :L, :]) and
                    torch.equal(cache_k.layers[i].values[..., :L, :], buf.base_v[i][..., :L, :])
                    for i in range(len(cache_k.layers)))
        view = buf.view(L4)
        trajs, fms = [], []
        for sd in SEEDS:
            torch.cuda.synchronize()
            tf = time.perf_counter()
            px = trajectory(model, prepared, raw, view, tk_4, target.eos, sd)[0]
            torch.cuda.synchronize()
            fms.append(round((time.perf_counter() - tf) * 1000, 3))
            trajs.append(np.asarray(px).reshape(64, 3))
        traj = np.stack(trajs)
        m4 = min_ade_fde(traj, gt)
        out = dict(kv_equal=bool(kv_eq), kv_len_k=Lk, kv_len_s4=L4, tokens_equal=(tk_4 == tk_k),
                   s4_minADE6_m=m4["minADE_m"], s4_minFDE6_m=m4["minFDE_m"], s4_traj=np.round(traj, 4).tolist(),
                   flow_ms_per_seed=fms)
        if not (kv_eq and tk_4 == tk_k):
            trajk = np.stack([np.asarray(trajectory(model, prepared, raw, cache_k, tk_k, target.eos, sd)[0]).reshape(64, 3)
                              for sd in SEEDS])
            mk = min_ade_fde(trajk, gt)
            out.update(k_minADE6_m=mk["minADE_m"], k_minFDE6_m=mk["minFDE_m"])
        del cache_k
    finally:
        kernels.uninstall(model)
    return out


class VisionTimer:
    """Times the vision encoder's forward between CUDA synchronizations."""

    def __init__(self, model) -> None:
        mods = [(n, m) for n, m in model.named_modules() if n.endswith("visual")]
        self.name, mod = mods[0]
        self.on, self.t = False, []
        mod.register_forward_pre_hook(self._pre)
        mod.register_forward_hook(self._post)

    def _pre(self, m, a):
        if self.on:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()

    def _post(self, m, a, o):
        if self.on:
            torch.cuda.synchronize()
            self.t.append((time.perf_counter() - self.t0) * 1000)

    def start(self) -> None:
        self.t, self.on = [], True

    def stop(self) -> list[float]:
        self.on = False
        return list(self.t)


@torch.no_grad()
def run_official(model, target, inp, vis, with6: bool) -> dict:
    """The public inference function with the arguments of its example (default kernels).

    The function changes model.vlm.generation_config in place; the caller restores it.
    """
    from respect.alpamayo.model import make_raw_inputs
    assert not kernels.is_installed()
    out = {}
    for ns in ((1, 6) if with6 else (1,)):
        raw = make_raw_inputs(target, inp)
        torch.cuda.manual_seed_all(42)
        vis.start()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=raw, top_p=0.98, temperature=0.6, num_traj_samples=ns, max_generation_length=256)
        torch.cuda.synchronize()
        out[f"traj{ns}"] = dict(total_ms=round((time.perf_counter() - t0) * 1000, 3),
                                vision_ms=[round(x, 3) for x in vis.stop()])
    return out


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
