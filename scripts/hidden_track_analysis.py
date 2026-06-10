#!/usr/bin/env python3
"""Per-attack EER on the eval vs hidden ASVspoof 2021 LA phases (Phase 2.2).

Characterises *where* the residual error lives. The hidden track is the hard,
OOD-like partition; this shows it's the *same* attacks (A07–A19) in a much harder
form, which is why augmentation/capacity don't fix it — an honest limit, stated
as a result rather than hidden.

Uses a SHARED per-phase bonafide pool for each attack's EER (partitioning the
real trials by attack would give n_real=0 → NaN; that was a real harness bug).

Consumes the cached scores .npz from `run_official_eval.py --save-scores`
(arrays: scores, labels, phases, attacks).

Usage:
    python3 scripts/hidden_track_analysis.py runs/scores_v9c_official.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from voiceguard.evaluation.metrics import compute_eer  # noqa: E402


def per_attack_eer(scores, labels, attacks, phase_mask) -> dict[str, float]:
    """EER per attack within a phase, each vs the phase's shared bonafide pool."""
    real = phase_mask & (labels == 0)
    real_scores = scores[real]
    out = {}
    for atk in sorted(set(attacks[phase_mask & (labels == 1)].tolist())):
        spoof = phase_mask & (labels == 1) & (attacks == atk)
        s = np.concatenate([real_scores, scores[spoof]])
        lb = np.concatenate([np.zeros(real_scores.size), np.ones(int(spoof.sum()))])
        out[atk] = round(compute_eer(s, lb) * 100, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    scores, labels = d["scores"].astype(float), d["labels"].astype(int)
    phases, attacks = d["phases"], d["attacks"]

    eval_m, hidden_m = phases == "eval", phases == "hidden"
    eval_eer = per_attack_eer(scores, labels, attacks, eval_m)
    hidden_eer = per_attack_eer(scores, labels, attacks, hidden_m)

    rows = []
    for atk in sorted(set(eval_eer) | set(hidden_eer)):
        e, h = eval_eer.get(atk), hidden_eer.get(atk)
        delta = round(h - e, 2) if (e is not None and h is not None) else None
        rows.append({"attack": atk, "eval_eer": e, "hidden_eer": h, "delta": delta})

    result = {"source": args.npz, "per_attack": rows}
    print("attack | eval EER | hidden EER | Δ(hidden-eval)")
    print("-------|---------:|-----------:|--------------:")
    for r in rows:
        e = f"{r['eval_eer']:.2f}%" if r["eval_eer"] is not None else "—"
        h = f"{r['hidden_eer']:.2f}%" if r["hidden_eer"] is not None else "—"
        dl = f"+{r['delta']:.2f}" if r["delta"] is not None else "—"
        print(f"{r['attack']:>6} | {e:>8} | {h:>10} | {dl:>13}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nSaved -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
