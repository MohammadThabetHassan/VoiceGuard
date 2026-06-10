#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals for EER + corrected minDCF from cached scores.

Phase 1.1 of docs/RESEARCH_RIGOR_PLAN.md. Consumes the .npz produced by
`run_official_eval.py --save-scores` (arrays: scores, labels, phases, attacks)
so CIs recompute in seconds without re-running inference.

Stratified bootstrap: reals and fakes are resampled with replacement separately
(preserving class counts), which is the appropriate scheme for an EER CI.

Usage:
    python3 scripts/bootstrap_ci.py runs/scores_v9c_official.npz --n 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from voiceguard.evaluation.metrics import compute_eer, compute_min_dcf  # noqa: E402


def eer_ci(scores: np.ndarray, labels: np.ndarray, n: int, seed: int = 0) -> dict:
    """Point EER + 95% CI via stratified bootstrap (percentile method)."""
    rng = np.random.default_rng(seed)
    real_idx = np.flatnonzero(labels == 0)
    fake_idx = np.flatnonzero(labels == 1)
    point = compute_eer(scores, labels) * 100
    boot = np.empty(n)
    for i in range(n):
        ri = rng.choice(real_idx, size=real_idx.size, replace=True)
        fi = rng.choice(fake_idx, size=fake_idx.size, replace=True)
        idx = np.concatenate([ri, fi])
        boot[i] = compute_eer(scores[idx], labels[idx]) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "eer": round(float(point), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "n_real": int(real_idx.size),
        "n_fake": int(fake_idx.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="cached scores .npz (scores,labels,phases[,attacks])")
    ap.add_argument("--n", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    scores, labels = d["scores"].astype(float), d["labels"].astype(int)
    phases = d["phases"] if "phases" in d else np.array(["all"] * len(scores))

    result = {
        "source": args.npz,
        "bootstrap_n": args.n,
        "min_dcf_corrected": round(float(compute_min_dcf(scores, labels)), 6),
        "full_pool": eer_ci(scores, labels, args.n),
        "per_phase": {},
    }
    for ph in sorted(set(phases.tolist())):
        m = phases == ph
        if labels[m].sum() and (1 - labels[m]).sum():
            result["per_phase"][ph] = eer_ci(scores[m], labels[m], args.n)

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"Saved -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
