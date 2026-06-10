# Hidden-Track Analysis — where the residual EER lives (Phase 2.2)

_Deployed model **v9c**, official ASVspoof 2021 LA. Reproducible from cached scores:_
`python3 scripts/hidden_track_analysis.py runs/scores_v9c_official.npz`

The headline 2.84% eval EER hides a sharp split by phase: full-pool EER is 8.21%
because the **hidden** track sits at **20.71%** while eval/progress are ~2.8%. This
is the dominant residual error, so it's worth characterising honestly rather than
burying it in a pooled number.

## Finding: it's the *same* attacks, much harder — not a coverage gap

Per-attack EER, each scored against the phase's shared bonafide pool (partitioning
reals by attack would give n_real=0 → NaN; that was a real harness bug, avoided here):

| Attack | eval EER | hidden EER | Δ (hidden − eval) |
|:------:|---------:|-----------:|------------------:|
| A07 | 1.51% | 23.78% | +22.27 |
| A08 | 1.95% | 14.03% | +12.08 |
| A09 | 0.41% | 11.17% | +10.76 |
| **A10** | 2.69% | **43.21%** | **+40.52** |
| **A11** | 1.88% | **35.10%** | **+33.22** |
| **A12** | 1.99% | **29.39%** | **+27.40** |
| A13 | 0.57% | 13.19% | +12.62 |
| A14 | 0.97% | 15.82% | +14.85 |
| A15 | 1.35% | 19.18% | +17.83 |
| A16 | 2.32% | 23.64% | +21.32 |
| A17 | 3.30% | 11.33% | +8.03 |
| A18 | 6.92% | 15.26% | +8.34 |
| A19 | 2.77% | 13.06% | +10.29 |

Every attack family (A07–A19) is present in *both* phases and is **far harder** in the
hidden one. The worst — A10/A11/A12 (neural TTS/VC) — blow up to 29–43% EER. Because the
hidden track is the same attacks under harder (unseen channel/codec/condition) settings,
**this is not something more training data of the same engines fixes**, and it is the
mechanism behind the KB's observed *anti-correlation*: augmentation/capacity that sharpen
the clean eval distribution (XLS-R+AASIST, RawBoost) tend to *degrade* the hidden track.

## Implication

- Quoting eval-only EER (2.84%) is fair for the standard partition but would overstate
  real-world robustness; the **full-pool 8.21%** is the honest headline, with this table
  explaining the gap.
- Closing the hidden track is a genuine open problem (channel/condition generalisation),
  not a quick fine-tune — consistent with [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md).
