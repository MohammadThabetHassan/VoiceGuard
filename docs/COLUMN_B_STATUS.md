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

### Result

_(appended after the run — see the closing section of this file.)_
