"""Gate 1 summary: accuracy tables, gains with CIs, Table S4 comparison, pass rule, drift, PC terms.

Every metric is recomputed from the saved predictions_sigma*.npz (and checked against
each run's own metrics.json). The pass rule (src/gate1.check_pass_rule) is applied to the
primary configuration (the first --runs folder: fixed + PCoderN); the others are reported
for comparison only. The choice of HP set / PCoder type is left to the user.

Usage (repo root):
  python scripts/summarize_gate1.py --runs <cfg1> <cfg2> <cfg3> <cfg4> --pc-terms <folder>
Writes results/gate1_summary_<timestamp>/{config.json, summary.json, summary.md, accuracy.csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gate1 import PAPER_TABLE_S4, SigmaRecords, check_pass_rule, sigma_metrics  # noqa: E402

DRIFT_T = (1, 2, 5, 10, 15)
DIAGNOSTICS = [  # order fixed by the user before the run
    "compare with configurations (2)-(4)",
    "runtime-calibrated CUDA C_sqrt values (diagnostic only)",
    "the App. A.1 alpha values",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True, help="config folders; first = primary")
    p.add_argument("--pc-terms", type=Path, default=None, help="results/gate1_pc_terms_* folder")
    p.add_argument("--out-root", type=Path, default=ROOT / "results")
    return p.parse_args()


def load_run(folder: Path) -> tuple[str, dict[float, SigmaRecords], dict]:
    """-> (name 'hp_pcoder', records per sigma, recomputed metrics per sigma)."""
    config = json.loads((folder / "config.json").read_text())
    saved = json.loads((folder / "metrics.json").read_text())
    records, metrics = {}, {}
    for sigma in config["sigmas"]:
        tag = f"{sigma:.2f}"
        z = np.load(folder / f"predictions_sigma{tag}.npz")
        records[sigma] = SigmaRecords(**{k: z[k] for k in z.files})
        metrics[sigma] = sigma_metrics(records[sigma], config["n_boot"])
        expected = {k: v for k, v in saved[tag].items() if k != "t0_vs_vgg16_max_abs_logit_diff"}
        assert metrics[sigma] == expected, f"{folder.name} sigma {tag}: recomputed metrics differ"
    return f"{config['hp_set']}_{config['pcoder_type']}", records, metrics


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def gain_str(g: dict) -> str:
    return f"{100 * g['gain']:+.2f} [{100 * g['ci95'][0]:+.2f}, {100 * g['ci95'][1]:+.2f}]"


def accuracy_section(runs: dict) -> list[str]:
    lines = ["## Top-1 accuracy (%) per t", ""]
    for name, (_, metrics) in runs.items():
        T = len(next(iter(metrics.values()))["top1"]) - 1
        lines += [f"### {name}", "", "| sigma | " + " | ".join(f"t={t}" for t in range(T + 1)) + " |",
                  "|---|" + "---|" * (T + 1)]
        lines += [f"| {s} | " + " | ".join(pct(a) for a in m["top1"]) + " |" for s, m in metrics.items()]
        lines.append("")
    return lines


def gain_section(runs: dict) -> list[str]:
    lines = ["## Gain t = 15 vs t = 0 (pp, paired bootstrap 95% CI) and best t", "",
             "Best t is chosen on the same images (in-sample), so its gain is optimistic.", "",
             "| config | sigma | acc t=0 | acc t=15 | gain t=15 [CI] | best t | gain best t [CI] |",
             "|---|---|---|---|---|---|---|"]
    for name, (_, metrics) in runs.items():
        for s, m in metrics.items():
            lines.append(f"| {name} | {s} | {pct(m['top1'][0])} | {pct(m['top1'][-1])} | {gain_str(m['gain_T'])} "
                         f"| {m['best_t']['t']} | {gain_str(m['best_t'])} |")
    return lines + [""]


def paper_section(runs: dict) -> list[str]:
    lines = ["## Comparison with Choksi 2021 Table S4 (PVGG16, fixed set)", "",
             "| sigma | paper t=0 | paper t=15 | paper gain | config | ours t=0 | ours t=15 | ours gain [CI] |",
             "|---|---|---|---|---|---|---|---|"]
    for s, (p0, p15) in PAPER_TABLE_S4.items():
        for name, (_, metrics) in runs.items():
            m = metrics[s]
            lines.append(f"| {s} | {p0:.2f} | {p15:.2f} | {p15 - p0:+.2f} | {name} | {pct(m['top1'][0])} "
                         f"| {pct(m['top1'][-1])} | {gain_str(m['gain_T'])} |")
    return lines + [""]


def rule_section(primary: str, verdicts: dict) -> list[str]:
    v = verdicts[primary]
    lines = [f"## Gate 1 pass rule (fixed before the run), applied to {primary}", "",
             "| rule | condition | values | result |", "|---|---|---|---|"]
    for key, r in v["rules"].items():
        lines.append(f"| ({key}) | {r['text']} | {json.dumps(r['detail'])} | {'PASS' if r['pass'] else 'FAIL'} |")
    lines += ["", f"**Configuration {primary}: Gate 1 {'PASSED' if v['pass'] else 'FAILED'}.**", ""]
    if not v["pass"]:
        failed = [k for k, r in v["rules"].items() if not r["pass"]]
        lines += [f"Failed rule(s): {', '.join(failed)}. The rule is not changed. Proposed diagnostics, in order:", ""]
        lines += [f"{i}. {d}" for i, d in enumerate(DIAGNOSTICS, start=1)] + [""]
    lines += ["Same rule for the other configurations (information only, not used for any choice):", ""]
    lines += [f"- {name}: " + ", ".join(f"({k}) {'pass' if r['pass'] else 'fail'}" for k, r in vv["rules"].items())
              for name, vv in verdicts.items() if name != primary]
    return lines + [""]


def pcoder_vs_pcodern(runs: dict, pc_terms: Path | None) -> tuple[list[str], dict]:
    """Observation: plain PCoder vs PCoderN with the same HP set."""
    lines = ["## Observation: PCoder vs PCoderN", ""]
    out = {}
    for hp in ("fixed", "tuned"):
        a, b = runs.get(f"{hp}_PCoderN"), runs.get(f"{hp}_PCoder")
        if a is None or b is None:
            continue
        acc_diff = max(abs(x - y) for s in a[1] for x, y in zip(a[1][s]["top1"], b[1][s]["top1"]))
        same_pred = {f"{s:.2f}": float((a[0][s].pred_top1[-1] == b[0][s].pred_top1[-1]).mean()) for s in a[0]}
        out[hp] = {"max_abs_top1_diff_pp": 100 * acc_diff, "same_top1_prediction_at_t15": same_pred}
        lines.append(f"- {hp}: max |top-1 difference| over all sigma and t = {100 * acc_diff:.2f} pp; "
                     "fraction of images with the same top-1 prediction at t = 15: "
                     + ", ".join(f"sigma {s}: {v:.4f}" for s, v in same_pred.items()))
    lines.append("")
    if pc_terms is not None:
        terms = json.loads((pc_terms / "terms.json").read_text())
        cfg = json.loads((pc_terms / "config.json").read_text())
        lines += [f"Why: size of the error term alpha * scale * grad relative to beta * ff "
                  f"(mean over {cfg['n_images']} images in one batch, from {pc_terms.name}; "
                  "scale = prd.numel()/C_sqrt for PCoderN, 1 for PCoder):", "",
                  "| sigma | config | t | " + " | ".join(f"pc{k}" for k in range(1, 6)) + " |",
                  "|---|---|---|" + "---|" * 5]
        for s, by_cfg in terms.items():
            for name, by_pc in by_cfg.items():
                for t in (1, 15):
                    lines.append(f"| {s} | {name} | {t} | "
                                 + " | ".join(f"{by_pc[f'pcoder{k}']['err_over_ff'][t - 1]:.1e}" for k in range(1, 6)) + " |")
        out["pc_terms_folder"] = pc_terms.name
        lines += ["", "With both variants the error term is orders of magnitude smaller than the "
                  "feedforward term (PCoderN about 1e-4 to 1e-3, PCoder about 1e-9 to 1e-7), so in "
                  "both the update is driven by the feedforward, feedback and memory terms; this is "
                  "why the two variants give nearly the same accuracy. Full term norms: "
                  f"{pc_terms.name}/terms.md.", ""]
    return lines, out


def drift_section(folder: Path) -> tuple[list[str], dict]:
    path = folder / "drift_summary.json"
    if not path.exists():
        return [], {}
    d = json.loads(path.read_text())
    lines = [f"## Feature drift ||rep(t) - rep(0)|| / ||rep(0)|| ({folder.name}, sigma = {d['sigma']}, "
             f"{d['n_images']} images)", "", "Mean (median) over images.", "",
             "| PCoder | " + " | ".join(f"t={t}" for t in DRIFT_T) + " |", "|---|" + "---|" * len(DRIFT_T)]
    for k in range(1, 6):
        s = d[f"pcoder{k}"]
        lines.append(f"| pc{k} | " + " | ".join(f"{s['mean'][t]:.4f} ({s['median'][t]:.4f})" for t in DRIFT_T) + " |")
    return lines + [""], {f"pcoder{k}": {str(t): d[f"pcoder{k}"]["mean"][t] for t in DRIFT_T} for k in range(1, 6)}


def write_csv(path: Path, runs: dict) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "sigma", "t", "top1", "top5"])
        for name, (_, metrics) in runs.items():
            for s, m in metrics.items():
                for t, (a1, a5) in enumerate(zip(m["top1"], m["top5"])):
                    w.writerow([name, s, t, f"{a1:.6f}", f"{a5:.6f}"])


def main() -> None:
    args = parse_args()
    runs = {}
    for folder in args.runs:
        name, records, metrics = load_run(folder)
        runs[name] = (records, metrics)
    primary = next(iter(runs))
    verdicts = {name: check_pass_rule(m) for name, (_, m) in runs.items()}

    obs_lines, obs = pcoder_vs_pcodern(runs, args.pc_terms)
    drift_lines, drift = drift_section(args.runs[0])
    lines = ["# Gate 1 summary: Predify Gaussian-noise gain on ImageNet val", "",
             f"Runs: {', '.join(f.name for f in args.runs)}. 2,000 Gate 1 images unless stated in the run config.", ""]
    lines += rule_section(primary, verdicts) + paper_section(runs) + gain_section(runs)
    lines += obs_lines + drift_lines + accuracy_section(runs)

    out_dir = args.out_root / f"gate1_summary_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"runs": [f.name for f in args.runs], "primary": primary,
         "pc_terms": args.pc_terms.name if args.pc_terms else None,
         "note": "metrics recomputed from predictions npz and checked equal to each run's metrics.json"},
        indent=2) + "\n")
    summary = {"primary": primary, "verdicts": verdicts, "pcoder_vs_pcodern": obs, "drift_mean": drift,
               "metrics": {name: {f"{s:.2f}": m for s, m in metrics.items()} for name, (_, metrics) in runs.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "summary.md").write_text("\n".join(lines))
    write_csv(out_dir / "accuracy.csv", runs)
    print("\n".join(lines[:60]))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
