# Adversarial Robustness — measured PGD/FGSM curve (Phase 2.1)

_Deployed model **v9c**, balanced sample (75 real / 75 fake) of the official
ASVspoof 2021 LA eval. Reproducible:_
`python3 scripts/pgd_curve.py --checkpoint runs/xlsr_aasist_v9c/model_best.pt --config runs/xlsr_aasist_v9c/config.json --flac-dir <eval>/flac`

This page **measures and characterises** the model's adversarial vulnerability
honestly. It is a stated limitation, not a solved problem — see the closing note.

## Threat model

- **Attacker capability:** white-box — full access to the model weights and gradients.
- **Perturbation:** L∞-bounded additive noise on the raw 16 kHz waveform, budget
  `epsilon` on amplitude (∈ [0, 1]); at the budgets below the perturbation is
  inaudible / near-inaudible.
- **Attacks:** single-step **FGSM** and 10-step **PGD** (step `alpha = epsilon/5`).
- **Out of scope here:** black-box / transfer attacks, physical-channel (over-the-air)
  attacks, and adaptive attacks against a defended model (no defense is claimed).

## Result — clean robustness, PGD-fragile

| epsilon | clean acc | FGSM acc | PGD acc |
|--------:|:---------:|:--------:|:-------:|
| 0.0005  | 0.980 | 0.940 | **0.507** |
| 0.001   | 0.980 | 0.913 | **0.167** |
| 0.002   | 0.980 | 0.873 | **0.020** |
| 0.005   | 0.980 | 0.880 | **0.000** |
| 0.010   | 0.980 | 0.900 | **0.007** |

- **Clean accuracy is 98%** and FGSM only dents it (~87–94%) — the model resists the
  cheap single-step attack.
- **PGD collapses it.** A tiny budget (ε=0.002, inaudible) drives accuracy to **2%**,
  and ε≥0.005 to **~0%**. The gap between FGSM and PGD shows the loss surface is
  locally smooth but globally exploitable by an iterative attacker.

## Interpretation & status (honest)

The fragility lives in the **frozen XLS-R backbone**, not the AASIST head. Per the KB,
head-only / partial-backbone adversarial fine-tunes **did not fix PGD** (0→4%) and
**failed the real-world gate** (fake-detect 90→33%) — so no hardened checkpoint is
deployed; production stays the clean-accurate v9c. A genuine fix needs **backbone
adversarial training**, which trades against clean EER and real-world robustness and is
scoped as a multi-week research item (`RESEARCH_RIGOR_PLAN.md` Column B / B.2), not a
quick patch.

**Bottom line:** VoiceGuard is robust to *natural* OOD (codecs, noise, unseen TTS) but is
**not** certified against a white-box adversarial attacker. This is disclosed, measured,
and reproducible rather than hidden — see also [`SECURITY.md`](../SECURITY.md).
