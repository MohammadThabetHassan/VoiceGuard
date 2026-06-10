# Clone-Detection Score Distributions (Phase 2.3)

_Deployed model **v9c** on the **speaker/text-disjoint** held-out set
(`heldout_eval_large`, 100 clips/family). Reproducible:_
`python3 scripts/clone_score_distributions.py --checkpoint runs/xlsr_aasist_v9c/model_best.pt --config runs/xlsr_aasist_v9c/config.json --eval-dir heldout_eval_large`

Rather than a single accuracy number, this reports the **fake-probability
distribution** per family — the honest way to show how close a clone family sits
to the real distribution (i.e. how near the front-end detectability limit).

| family | n | mean | median | p10–p90 | detect / pass |
|--------|:-:|:----:|:------:|:-------:|:-------------:|
| real | 100 | 0.08 | 0.02 | 0.02–0.10 | **96.0%** pass |
| XTTS | 100 | 0.98 | 0.99 | 0.98–0.99 | **100.0%** detect |
| IndexTTS-2 | 100 | 0.96 | 0.99 | 0.97–0.99 | **97.0%** detect |

## What the distributions show

- **Clean separation.** Real audio clusters near 0 (median 0.02) and both clone
  families near 1 (median 0.99). The score distributions barely overlap — there is
  no ambiguous middle, so the verdicts are confident, not marginal.
- **IndexTTS-2 is *not* at the ceiling for v9c.** Earlier models (v3/v6) caught fresh
  IndexTTS-2 at only ~60% — that was the "front-end ceiling" claim. The v9c recipe
  (diverse clones cloned from ASVspoof-real speakers, ASVspoof anchor, top-12 layers
  unfrozen) **disproves it on this held-out set**: 97% detect at median fake-prob 0.99.
  The earlier `KNOWN_LIMITATIONS.md` ceiling language applies to the *superseded*
  models, not v9c.
- **Residual error is small and tail-shaped.** The 3% missed IndexTTS-2 and 4%
  false-flagged real are distribution tails, not a systematic gap.

## Honest caveats

- This is one held-out corpus (XTTS / IndexTTS-2 / LibriSpeech-derived reals); it is
  **not** an exhaustive sweep of every TTS engine or recording channel. Premium
  commercial engines beyond the trained set can still be OOD (see ElevenLabs in
  `RESULTS.md`).
- These are **held-out clone** metrics, separate from the official ASVspoof EER
  ([`RESULTS_canonical.md`](RESULTS_canonical.md)) — the two protocols are not mixed.
