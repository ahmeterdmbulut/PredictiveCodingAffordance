"""Run the four Gate 1 configurations in order, each in its own process and results folder.

Order: (1) fixed + PCoderN (primary, with drift), (2) tuned + PCoderN, (3) fixed + PCoder,
(4) tuned + PCoder. A crash in one configuration is logged and the next one still runs.
On Windows the machine is kept from idle-sleeping while this process lives
(SetThreadExecutionState; released automatically when it exits).

Usage (repo root):  python scripts/run_gate1_grid.py [--device cuda] [--n-images N]
Logs: results/gate1_grid_<timestamp>/<config>.log and status.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [  # (hp_set, pcoder_type, drift)
    ("fixed", "PCoderN", True),
    ("tuned", "PCoderN", False),
    ("fixed", "PCoder", False),
    ("tuned", "PCoder", False),
]


def keep_awake() -> None:
    """Windows only: ask the OS not to idle-sleep while this process runs."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-images", type=int, default=None)
    args = p.parse_args()
    keep_awake()
    log_dir = ROOT / "results" / f"gate1_grid_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir.mkdir(parents=True)
    status = []
    for hp_set, pcoder_type, drift in CONFIGS:
        cmd = [sys.executable, str(ROOT / "scripts/run_gate1.py"), "--hp-set", hp_set,
               "--pcoder-type", pcoder_type, "--device", args.device]
        cmd += ["--drift"] if drift else []
        cmd += ["--n-images", str(args.n_images)] if args.n_images else []
        name = f"{hp_set}_{pcoder_type}"
        start = time.time()
        with (log_dir / f"{name}.log").open("w") as log:
            code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        status.append({"config": name, "returncode": code, "minutes": round((time.time() - start) / 60, 1),
                       "finished": datetime.now().isoformat(timespec="seconds")})
        (log_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(status[-1], flush=True)


if __name__ == "__main__":
    main()
