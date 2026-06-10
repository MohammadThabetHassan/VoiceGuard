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

### Result — IMPLEMENTED & PRE-REGISTERED; execution GPU-blocked (2026-06-11)

The runner (`voiceguard-checkpoints/column_b2_adv.py`) is complete: it loads v9c,
unfreezes the top-6 of 24 XLS-R encoder layers + the AASIST head, builds an
ASVspoof-bonafide real pool + clone fakes, and does clean+PGD adversarial training
with the gate measured afterwards by `scripts/pgd_curve.py`,
`scripts/clone_score_distributions.py`, and `scripts/eval_official.sh`.

A smoke test validated the full code path **up to the backward pass** (data decode,
model load, forward, FGSM/PGD forward all run). It then hit **CUDA OOM with only
~80 MiB free** — the GPU is held at **30.6 / 31 GB by another user's process**
(`ollama/llama-server`). Backbone adversarial training needs several GB of gradient
memory, so the run **cannot proceed under this contention**.

**Honest status:** this is an *execution* block (shared-GPU contention), not a code or
design block. The experiment is pre-registered and ready; it runs as soon as the GPU
frees. The expected outcome (per the head-only precedent + the disclosed gone-LibriSpeech
confound) is a robustness/accuracy trade that likely **fails the deploy gate** — in which
case v9c stays and that is the reported finding. No result is fabricated in the meantime.
