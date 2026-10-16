"""Module and GPU power sampling on Jetson AGX Thor through the INA sensors in sysfs."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from statistics import mean


def _hwmon(name: str) -> Path | None:
    for d in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (d / "name").read_text().strip() == name:
                return d
        except OSError:
            continue
    return None


_INA238, _INA3221 = _hwmon("ina238"), _hwmon("ina3221")
MODULE_POWER = (_INA238 / "power1_input") if _INA238 else Path("/nonexistent")  # uW, whole module
GPU_CURR = (_INA3221 / "curr1_input") if _INA3221 else Path("/nonexistent")     # mA, VDD_GPU
GPU_VOLT = (_INA3221 / "in1_input") if _INA3221 else Path("/nonexistent")       # mV, VDD_GPU


class PowerSampler(threading.Thread):
    """Background sampler of (time, module W, GPU W)."""

    def __init__(self, period_s: float = 0.02) -> None:
        super().__init__(daemon=True)
        self.period_s = period_s
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                mod_w = int(MODULE_POWER.read_text()) / 1e6
                gpu_w = (int(GPU_CURR.read_text()) / 1e3) * (int(GPU_VOLT.read_text()) / 1e3)
                self.samples.append((time.perf_counter(), mod_w, gpu_w))
            except Exception:
                pass
            self._stop.wait(self.period_s)

    def stop(self) -> None:
        self._stop.set()

    def window(self, t0: float, t1: float) -> dict[str, float] | None:
        rows = [s for s in self.samples if t0 <= s[0] <= t1]
        if len(rows) < 2:
            return None
        return {"n_samples": len(rows), "module_w": round(mean(r[1] for r in rows), 3),
                "gpu_w": round(mean(r[2] for r in rows), 3)}
