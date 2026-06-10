# Evaluation Protocols & Reproducibility (Phase 1.4 + 1.5)

Two different "EER rulers" appear across VoiceGuard's history. They are **not
comparable**, and conflating them is the single biggest rigor trap. This page
states which is authoritative and how to reproduce every headline number.

## The two rulers

| Ruler | What it is | Class balance | Use |
|-------|-----------|:-------------:|-----|
| **Official ASVspoof 2021 LA** | The full 181,566-trial eval, raw FLAC, official CM keys | ~90% spoof / 10% bonafide | **Authoritative.** Every headline EER (e.g. v9c **2.84%**) is on this. Comparable to the literature. |
| **Balanced mirror** | `MoaazTalab/ASVspoof_2021_LA_Balanced_Normalized` (a re-balanced, RMS-normalised HF subset) | ~50/50 | **Internal proxy only.** Used during robustness iteration *while the official data was off-disk*. |

### Why they diverge so much

The same model reads very differently on the two: e.g. v7 is **3.38%** on the
official eval but **9.25%** on the balanced mirror; the early robustness lineage
(v3/v6) sat at ~29–34% on the mirror. The drivers:

1. **Class balance.** EER is threshold-free but still depends on the score
   distribution; a 50/50 set vs a 90/10 set place the equal-error operating point
   in different regions.
2. **Different, harder slice.** The mirror is normalised and re-sampled, not the
   official trial list — a different (often harder) distribution.
3. **Not the official trial protocol**, so it is not comparable to any published
   ASVspoof number.

### The rule

- **All headline / paper / README EERs use the official protocol.** The deployed
  model is **v9c = 2.84% [2.67–3.02]** there.
- The balanced-mirror EER is only ever valid for comparing models **against each
  other on that same mirror** during iteration — never quoted as "the EER", never
  compared to the official 2.84% or to the literature.
- v9c's balanced-mirror EER was **not** separately measured because it is not a
  reported metric; the official protocol is the sole headline ruler.

## Reproducing every number

Everything below regenerates from artifacts already on disk; no number is hand-typed.

```bash
# 1. Official eval for any checkpoint (writes JSON + caches raw scores)
scripts/eval_official.sh xlsr_aasist_v9c
#    -> runs/official_xlsr_aasist_v9c.json   (EER, per-phase)
#    -> runs/scores_xlsr_aasist_v9c_official.npz   (raw scores for fast recompute)
#    -> runs/ci_xlsr_aasist_v9c_official.json   (95% bootstrap CIs + corrected minDCF)

# 2. Bootstrap 95% CIs + corrected minDCF from cached scores (seconds, no GPU)
python3 scripts/bootstrap_ci.py runs/scores_v9c_official.npz --n 1000

# 3. Per-attack hidden-track analysis from cached scores (no GPU)
python3 scripts/hidden_track_analysis.py runs/scores_v9c_official.npz

# 4. Adversarial PGD/FGSM curve
python3 scripts/pgd_curve.py --checkpoint runs/xlsr_aasist_v9c/model_best.pt \
    --config runs/xlsr_aasist_v9c/config.json --flac-dir <eval>/flac

# 5. Regenerate the canonical results table from all official_*.json
VG_RUNS_DIR=<runs> python3 scripts/build_canonical_results.py > docs/RESULTS_canonical.md
```

### Determinism notes

- **Eval is deterministic** (fixed 3 s clip, `softmax[:,1]`, no sampling) — EER
  reproduces to four decimals (verified: v7/v8/v11/v12/parent re-scored with the
  fixed metrics code matched their published EER exactly).
- **Bootstrap CIs** use a fixed seed (`np.random.default_rng(0)`), so CIs are
  reproducible run-to-run.
- **PGD curve** sample selection is deterministic (first N per class from the eval
  phase); attack is gradient-based on fixed inputs.
- **Data provenance:** official eval at `asvspoof2021_LA_official/` (Zenodo
  `4837263` + asvspoof.org CM keys); the harness is vendored at
  `scripts/run_official_eval.py` so a clean clone is self-contained.
