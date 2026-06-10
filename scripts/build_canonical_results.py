#!/usr/bin/env python3
"""Build the canonical VoiceGuard results table from official-eval JSON artifacts.

Phase 0 of docs/RESEARCH_RIGOR_PLAN.md: ONE provenance-tagged results table,
generated programmatically so no number is hand-typed. Every row cites the JSON
it came from, which in turn was produced by `run_official_eval.py` on the
official ASVspoof 2021 LA protocol.

Usage:
    VG_RUNS_DIR=/srv/thabet/voiceguard-checkpoints/runs \
        python3 scripts/build_canonical_results.py > docs/RESULTS_canonical.md
"""

from __future__ import annotations

import glob
import json
import os
from datetime import date

RUNS_DIR = os.environ.get("VG_RUNS_DIR", "/srv/thabet/voiceguard-checkpoints/runs")

# Production-lineage checkpoints: role + ordering (lower sort_key = higher).
# Keeps superseded/regressed runs VISIBLE — that transparency is the point.
ROLES: dict[str, tuple[int, str]] = {
    "v9c": (0, "🏆 deployed (production)"),
    "v7": (1, "prior production"),
    "parent": (2, "EER-only (clone-blind)"),
    "v8": (3, "EER-opt experiment (clone-regressed)"),
    "v11": (4, "superseded — regressed"),
    "v12": (5, "superseded — regressed"),
}

# Same-protocol architecture baselines: run-name substring -> (display, order).
BASELINES: dict[str, tuple[str, int]] = {
    "wav2vec2-large": ("Wav2Vec2-large", 0),
    "wavlm-base-plus": ("WavLM-base-plus", 1),
    "wavlm-large": ("WavLM-large", 2),
    "_aasist_aug": ("XLS-R + AASIST (aug)", 3),
}


def short_label(path: str) -> str:
    name = os.path.basename(path).removeprefix("official_").removesuffix(".json")
    return name.removeprefix("xlsr_aasist_")


def classify(run: str, label: str) -> tuple[str, int, str, str]:
    """Return (kind, sort_key, display_label, role). kind in {lineage, baseline}."""
    if label in ROLES:
        sk, role = ROLES[label]
        return ("lineage", sk, label, role)
    for sub, (disp, order) in BASELINES.items():
        if sub in run:
            return ("baseline", order, disp, "SSL baseline")
    return ("baseline", 99, label, "SSL baseline")


def _load(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "official_*.json"))):
        if path.endswith("_recompute.json"):
            continue  # merged into its base row below
        run = os.path.basename(path).removeprefix("official_").removesuffix(".json")
        d = json.load(open(path))  # noqa: SIM115
        kind, sort_key, label, role = classify(run, short_label(path))
        full = d.get("headline_full_pool", {})
        phase = d.get("per_phase_eer", {})

        # minDCF: prefer a fixed-metrics recompute; baselines are freshly scored
        # with the corrected code so their base file is already right; lineage
        # base files predate the fix, so require a recompute for them.
        recompute = _load(os.path.join(RUNS_DIR, f"official_{run}_recompute.json"))
        if recompute:
            min_dcf = recompute["headline_full_pool"]["min_dcf"]
        elif kind == "baseline":
            min_dcf = full.get("min_dcf")
        else:
            min_dcf = None

        ci = _load(os.path.join(RUNS_DIR, f"ci_{run}_official.json"))
        eval_ci = ci["per_phase"].get("eval", {}).get("ci95") if ci else None

        rows.append(
            {
                "kind": kind,
                "label": label,
                "role": role,
                "sort_key": sort_key,
                "eval": phase.get("eval", {}).get("eer"),
                "eval_ci": eval_ci,
                "progress": phase.get("progress", {}).get("eer"),
                "hidden": phase.get("hidden", {}).get("eer"),
                "full_pool": full.get("eer"),
                "min_dcf": min_dcf,
                "checkpoint": d.get("checkpoint", "?"),
                "source": os.path.basename(path),
            }
        )
    rows.sort(key=lambda r: (r["kind"] != "lineage", r["sort_key"], r["label"]))
    return rows


def fmt(x: float | None, suffix: str = "%") -> str:
    return f"{x:.2f}{suffix}" if isinstance(x, (int, float)) else "—"


def _render_table(rows: list[dict]) -> None:
    cols = ["Model", "Role", "eval EER [95% CI]", "progress EER", "hidden EER",
            "full-pool EER", "minDCF"]
    print("| " + " | ".join(cols) + " |")
    print("|-------|------|:-----------------:|:------------:|:----------:|:-------------:|:------:|")
    for r in rows:
        if r["eval_ci"]:
            lo, hi = r["eval_ci"]
            eval_str = f"**{fmt(r['eval'])}** [{lo:.2f}–{hi:.2f}]"
        else:
            eval_str = fmt(r["eval"])
        dcf = r["min_dcf"]
        dcf_str = f"{dcf:.3f}" if isinstance(dcf, (int, float)) else "—"
        print(
            f"| **{r['label']}** | {r['role']} | {eval_str} | {fmt(r['progress'])} "
            f"| {fmt(r['hidden'])} | {fmt(r['full_pool'])} | {dcf_str} |"
        )


def main() -> None:
    rows = load_rows()
    lineage = [r for r in rows if r["kind"] == "lineage"]
    baselines = [r for r in rows if r["kind"] == "baseline"]

    print("# VoiceGuard — Canonical Results (single source of truth)\n")
    print(
        f"_Auto-generated by `scripts/build_canonical_results.py` on {date.today()} "
        f"from `{RUNS_DIR}/official_*.json`. Do not hand-edit — re-run the script._\n"
    )
    print(
        "All EERs are on the **official ASVspoof 2021 LA** protocol "
        "(181,566 trials; `run_official_eval.py`, 3 s clips, per-phase split). "
        "`eval EER` shows the 95% bootstrap CI where computed "
        "(`scripts/bootstrap_ci.py`). minDCF uses the corrected estimator "
        "(`p_target=0.05`); rows not yet recomputed show `—`.\n"
    )

    print("## Production lineage (XLS-R + AASIST)\n")
    _render_table(lineage)

    if baselines:
        print("\n## Same-protocol baselines\n")
        print(
            "**In-house reproductions** of standard SSL anti-spoofing architectures "
            "(our own trainings, *not* the literature's published numbers), scored on "
            "the **identical** eval so VoiceGuard's numbers are anchored to a common ruler.\n"
        )
        _render_table(baselines)

    print("\n## Provenance\n")
    print("| Model | Checkpoint | Source JSON |")
    print("|-------|------------|-------------|")
    for r in rows:
        print(f"| {r['label']} | `{r['checkpoint']}` | `{r['source']}` |")

    print(
        "\n## Notes\n"
        "- **eval** is the standard ASVspoof21-LA partition; **hidden** is the hard "
        "OOD-like track that resists augmentation/capacity (residual error concentrates here).\n"
        "- Clone-family detection and real-pass are measured on the **held-out** "
        "(speaker/text-disjoint) set, *not* this official protocol — they are reported "
        "separately and must not be mixed into this table's columns.\n"
        "- minDCF was corrected in Phase 0.3 (an inverted cost model previously pinned "
        "it ≈1.0); 95% bootstrap EER CIs are from `scripts/bootstrap_ci.py`.\n"
    )


if __name__ == "__main__":
    main()
