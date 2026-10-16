#!/usr/bin/env python3
"""Print the software and board state that the paper's measurements depend on.

Compares each item with the tested environment and flags differences. Output
tokens do not depend on clocks, but Decode times do: fix the clocks with
`sudo jetson_clocks` and use the 120 W power mode before timing runs.
"""
from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from pathlib import Path

TESTED = {
    "python": "3.12.13",
    "torch": "2.8.0a0+gitba56102",
    "torch.cuda": "13.0",
    "triton": "3.7.1",
    "transformers": "4.57.1",
    "alpamayo1_5 commit": "2eff7037e47afb96a578b3d1bca453a373cd781e",
    "L4T": "R39 (release), REVISION: 2.0",
    "power mode": "120W",
}


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def version(mod: str) -> str:
    try:
        return importlib.import_module(mod).__version__
    except Exception:
        return "not installed"


def main() -> None:
    env = {"python": platform.python_version(), "torch": version("torch"), "triton": version("triton"),
           "transformers": version("transformers")}
    try:
        import torch
        env["torch.cuda"] = torch.version.cuda or "none"
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device"
    except Exception:
        env["torch.cuda"] = "unavailable"
    try:
        import alpamayo1_5
        root = Path(alpamayo1_5.__file__).resolve().parents[2]
        env["alpamayo1_5 commit"] = sh(f"git -C {root} rev-parse HEAD") or "unknown"
    except Exception:
        env["alpamayo1_5 commit"] = "not installed"
    tegra = Path("/etc/nv_tegra_release")
    env["L4T"] = tegra.read_text().splitlines()[0].lstrip("# ").split(", GCID")[0] if tegra.exists() else "not a Jetson"
    mode = sh("nvpmodel -q").splitlines()
    env["power mode"] = mode[0].replace("NV Power Mode: ", "") if mode else "unknown"
    clocks = sh("sudo -n jetson_clocks --show")
    env["clocks"] = "locked (jetson_clocks)" if _locked(clocks) else "not verified; run `sudo jetson_clocks`"
    hub = Path.home() / ".cache/huggingface/hub"
    for repo in ("nvidia--Alpamayo-1.5-10B", "z-lab--Alpamayo-1.5-10B-DFlash"):
        env[f"weights {repo.replace('--', '/')}"] = "found" if (hub / f"models--{repo}").exists() else "missing"

    width = max(len(k) for k in env)
    for k, v in env.items():
        mark = ""
        if k in TESTED and not str(v).startswith(TESTED[k]):
            mark = f"   <- tested with {TESTED[k]}"
        print(f"{k:<{width}}  {v}{mark}")


def _locked(show: str) -> bool:
    """True when every GPU and memory clock domain has MinFreq == MaxFreq."""
    found = False
    for line in show.splitlines():
        if line.startswith(("gpu-", "EMC")) and "MinFreq=" in line:
            parts = dict(p.split("=", 1) for p in line.split() if "=" in p)
            found = True
            if parts.get("MinFreq") != parts.get("MaxFreq"):
                return False
    return found


if __name__ == "__main__":
    sys.exit(main())
