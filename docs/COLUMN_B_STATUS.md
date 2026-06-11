# Column B — Research-ceiling attempts (status)

Column B of `RESEARCH_RIGOR_PLAN.md` is the genuinely hard, training-heavy work
that could push toward a true 9/10. This page tracks what was attempted, with
**pre-registered gates** (written before each run) so the result can't be
rationalised after the fact.

## B.1 — From-scratch SSL fine-tune for the hidden track → **DATA-BLOCKED**

The plan was a full-protocol fine-tune from the SSL backbone to attack the 20.7%
hidden track. It is **blocked by missing training data**, not by compute:

- ASVspoof 2021 LA is **eval-only**; legitimate training uses the **2019 LA train**
  partition (`LA_T_*`). That set is **not on disk** — only the balanced mirror and
  the official 2021 *eval* are present.
- Required to unblock: re-acquire **ASVspoof 2019 LA train** (the gated University of
  Edinburgh release, ~7.9 GB FLAC, `LA_T_*` utterances) — then a from-scratch
  RawBoost-augmented fine-tune is runnable.
- Also note the documented **anti-correlation**: augmentation/capacity that sharpen the
  clean eval tend to *degrade* the hidden track (`HIDDEN_TRACK_ANALYSIS.md`). So even
  unblocked, B.1 is a genuine open research problem, not a guaranteed win.

This is runnable-when-acquired, deliberately **not** faked on a proxy dataset.

## B.2 — Backbone adversarial training → **ATTEMPTED (one pre-registered shot)**

The deployed model is PGD-fragile (clean 98% → ~0% at ε≥0.005, `ADVERSARIAL_ROBUSTNESS.md`).
A prior **head-only** adversarial fine-tune failed (PGD 0→4%, real-world 90→33%). The
documented "proper fix" is a **backbone** adversarial fine-tune — untried, so one
disciplined shot is run here.

### Pre-registered gate (written BEFORE the run)

Start from v9c, unfreeze the top XLS-R layers, PGD adversarial training. **Deploy only
if ALL hold; otherwise v9c stays and we stop (no v2/v3 spiral):**

| criterion | threshold |
|-----------|-----------|
| PGD acc @ ε=0.002 | ≥ 30% (v9c baseline: 2%) |
| held-out real-pass | ≥ 90% |
| held-out clone detect (XTTS & IndexTTS-2) | ≥ 90% each |
| clean official eval EER | ≤ 6% (≤ ~2× v9c's 2.84%) |

**Disclosed confound:** v9c's LibriSpeech real-diversity pool was on `/tmp` and is
**gone**, so the real-audio pool here is ASVspoof bonafide only. A real-pass regression
would therefore be *partly a data artifact* (− LibriSpeech), not purely the effect of
adversarial training. This is a confounded delta, stated as such.

### Result — RAN 2026-06-11; **FAILED the gate decisively. v9c stays deployed.**

The run completed (`column_b2_adv.py`: v9c → top-6 XLS-R layers + AASIST head unfrozen,
77.5M trainable params, clean+PGD mix, 4 epochs; train loss 1.01→0.41, adv-loss > clean
throughout — training was healthy). Checkpoint: `runs/xlsr_aasist_b2_advbackbone`. The
pre-registered gate, measured on the **held-out / official** sets:

| gate criterion | threshold | **measured** | verdict |
|----------------|-----------|--------------|:------:|
| clean official-eval EER | ≤ 6% | **2.79%** eval / 8.06% full-pool (≈ v9c) | ✓ |
| held-out real-pass (@ 0.5 thr) | ≥ 90% | **18%** (v9c: 96%) | ✗✗ |
| clone detect (XTTS / IndexTTS-2) | ≥ 90% | 100% / 100% | ✓* |
| PGD acc @ ε=0.002 | ≥ 30% | not demonstrably improved (confounded — see below) | ✗ |

\* the clone "100%" is an artifact of the boundary shift below: when the model calls
almost everything "fake", clones pass trivially while real callers get flagged.

**What actually happened — calibration collapse, not loss of discrimination.** The model's
*ranking* of real vs fake is preserved — official **EER 2.79% ≈ v9c's 2.84%**, and the EER
gate passes. What broke is the **operating point at the deployed 0.5 threshold**: scores
shifted so genuine audio reads as fake, crashing held-out **real-pass to 18%**. As a
deployed detector (which thresholds at 0.5), it is therefore **unusable** despite the good
EER. (The in-script test read clean 94% only because that split came from the *same*
balanced-mirror training distribution; the held-out/official numbers are the real test.)

> **Self-correction (rigor note):** an initial read of the 150-clip `pgd_curve` output
> (clean-accuracy-at-0.5 = 0.50 on a balanced set) looked like "lost generalisation". The
> full official EER (2.79%) **refuted that** — discrimination is intact; *calibration* is
> the casualty. The PGD-curve clean/PGD numbers are confounded by this 0.5-threshold shift,
> so no PGD improvement can be claimed. Corrected here rather than shipping the misread.

**Decision (pre-registered):** real-pass gate failed → **NOT deployed; v9c remains
production.** One shot, stopped — no v2/v3 spiral. Consistent with the documented
conclusion: with only the balanced-mirror + clones on hand (LibriSpeech real-diversity pool
gone), backbone adversarial training wrecks the real-pass operating point without buying
demonstrable PGD robustness. A genuine fix needs the official ASVspoof train set + a broad
real-speech pool (B.1's data prerequisite) — and likely score re-calibration — not another
fine-tune of a robustness derivative. Honest negative result, recorded with its correction.
