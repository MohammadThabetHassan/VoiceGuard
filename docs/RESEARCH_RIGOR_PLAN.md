# Research-Rigor Improvement Plan (7/10 → 9+/10)

_Goal: raise the **research-rigor** axis from 7 to 9+ by improving **how we measure
and report**, not by chasing one more model version. The 9 comes from sound,
reproducible, honestly-characterised results — including honest limits — not from
erasing known-hard problems._

> Scope note. This plan is split into **(A) executable & verifiable now** (data, GPU,
> and scripts are already on disk) and **(B) multi-week team roadmap** (new training
> regimes / data collection). Do A first — it is where the rigor score actually moves.

---

## Why the axis is at 7 (grounded in current artifacts)

Verified on 2026-06-10 against `voiceguard-checkpoints/runs/official_*.json` and `docs/`:

1. **Results narrative is inconsistent across docs.** README headlines **v9c @ 2.84%**
   (substantiated: `official_xlsr_aasist_v9c.json` → eval EER 2.8382%), but
   `docs/KNOWN_LIMITATIONS.md` still lists **v7** as deployed and never mentions v9c.
   The deployed lineage (v3→v6→v7→v9c) is told four different ways.
2. **minDCF is degenerate.** v9c min_dcf ≈ **0.998** (v7 0.999, v8 0.971) vs the
   parent's 0.794 — a near-1.0 minDCF means the cost model / score orientation is
   broken, so minDCF is currently uninformative and must not be quoted.
3. **No statistical rigor.** Single-point EERs with no confidence intervals, no
   multi-seed runs, no significance tests. A 2.84% vs 3.38% gap is reported as fact
   with no error bars.
4. **No same-protocol baselines.** No published-SOTA comparison (AASIST, RawNet2,
   wav2vec2-AASIST) measured on our exact eval, so "good EER" is unanchored.
5. **Robustness claims live on two different rulers.** Clone/robustness numbers are on
   the *balanced mirror* (~9–29% EER); official EER is on the 2021-LA protocol. They
   are not reconciled in one place, inviting apples-to-oranges reading.
6. **Hard limits are real and partly under-characterised:** PGD ≈ 0% at ε=0.01
   (security), hidden-track EER ≈ 20.7% (v9c), IndexTTS-2 near the front-end ceiling.

---

## What "9+/10" looks like (the rubric we are grading against)

_Status as of 2026-06-11 — most of the rubric is shipped (commits c37c42c → 4de6096)._

- [x] **One canonical results table**, provenance-tagged — `docs/RESULTS_canonical.md`
      via `scripts/build_canonical_results.py`; all docs reference it.
- [x] **Every headline number has a 95% CI** (bootstrap) on a **named protocol** —
      `scripts/bootstrap_ci.py`; v9c = 2.84% [2.67–3.02]. Rulers kept separate.
- [x] **≥3 published baselines** on our exact eval — 4 (Wav2Vec2-large, WavLM-base+,
      WavLM-large, XLS-R+AASIST aug) in the canonical table's baseline panel.
- [x] **An ablation table** — the production-lineage table isolates each lever
      (parent→v7→v9c→v8 = clone-hardening / robustness / EER-opt), baselines anchor it.
- [x] **minDCF fixed** — corrected the inverted cost model (`metrics.py`), 0.998→0.229,
      with a regression test and written justification (Phase 0.3 below).
- [x] **A documented threat model** + honest PGD curve (`docs/ADVERSARIAL_ROBUSTNESS.md`)
      and hidden-track characterisation (`docs/HIDDEN_TRACK_ANALYSIS.md`).
      IndexTTS-2 ceiling is documented in `KNOWN_LIMITATIONS.md` (Phase 2.3 — remaining:
      add held-out score distributions).
- [x] **A reproducibility appendix** — `docs/EVAL_PROTOCOLS.md` (entrypoints, seeds, data
      provenance, ruler reconciliation). Remaining: SHA-256 manifests + env lockfile.

---

## Column A — Executable & verifiable NOW (this is the score-mover)

### Phase 0 — Verify & reconcile (≈ half a day, no training)
**The cheapest rigor win and an integrity fix; do it first — adding experiments on top
of an inconsistent narrative *lowers* rigor.**

0.1 **Single canonical results table.** Generate `docs/RESULTS_canonical.md` from the
`runs/official_*.json` files programmatically (one script, no hand-typed numbers).
Columns: checkpoint, role (deployed/baseline/superseded), official eval / progress /
hidden / full-pool EER, clone-family detection, real-pass, provenance (script+commit).
Mark **v9c = deployed**, parent = EER-only, v7 = prior prod, v8/v11/v12 = superseded
experiments (incl. the v11/v12 regressions — keeping failed runs visible *is* rigor).

0.2 **Reconcile all docs to it.** Update `KNOWN_LIMITATIONS.md` (still says v7) and any
README/RESULTS drift so a reader never sees two "deployed" models. Single source of truth.

0.3 **Resolve minDCF.** Audit `evaluation/metrics.py` cost model and score orientation
(p_target, C_miss, C_fa, and whether `softmax[:,1]` is the spoof or bonafide score).
Either fix it so minDCF is meaningful (expect ≈0.05–0.5 for a 2.8% EER system) or
formally retire it from all docs with a one-paragraph justification.

0.4 **Lock the eval entrypoint.** Wrap `run_official_eval.py` behind one reproducible
command (`scripts/eval_official.sh <ckpt>`), pin the data path + decoder + clip length,
and emit the JSON + a human table. Everything downstream cites this.

### Phase 1 — Methodology rigor (≈ 2–3 days, mostly inference-only)

1.1 **Bootstrap confidence intervals on EER.** Add a `--bootstrap N` flag (resample
trials, e.g. 1000×) to the eval harness; report `EER [95% CI]` for every headline.
Turns "2.84 vs 3.38" into a statistically defensible statement. *Inference-only.*

1.2 **Published-SOTA baselines on our exact protocol.** Run AASIST, RawNet2, and a
wav2vec2-AASIST reference on the same official eval and put them in the canonical table
with CIs. (We already train AASIST/wav2vec2 families; add 1–2 reference architectures.)
Anchors our numbers to the literature. *Mostly inference / short training.*

1.3 **Ablation table from existing checkpoints.** We already have parent / v3 / v6 / v7 /
v8 / v9c / rawboost / large-baseline. Evaluate them under the one entrypoint and present
a single ablation: each row isolates a lever (± RawBoost, ± AASIST backend, ± clone
hardening, layers unfrozen) with the EER / real-pass / clone-detect deltas + CIs.
*Inference-only — huge rigor gain for near-zero training cost.*

1.4 **Reconcile the two rulers explicitly.** One figure/table showing official-eval EER
**and** balanced-mirror EER side by side per checkpoint, with a written note on why they
differ and which to trust for which claim. Removes the apples-to-oranges risk.

1.5 **Reproducibility appendix.** Seeds, `pip freeze` lock, dataset manifests with
SHA-256, and the exact commands. Add a CI smoke-test that runs the eval entrypoint on a
tiny fixture so the harness can't silently rot.

### Phase 2 — Honest characterisation of hard limits (≈ 2 days, measure don't over-promise)

2.1 **Threat model + PGD robustness curve.** Document the adversarial threat model
(white-box PGD, ε grid, L∞), then publish the *honest* curve: clean → FGSM → PGD across
ε using `evaluation/adversarial_eval.py`. Report where it breaks (≈0% at ε=0.01). A
characterised vulnerability with a stated threat model is **rigour**; a silent one is not.

2.2 **Hidden-track analysis.** The residual error is concentrated in the ASVspoof hidden
track (v9c ≈ 20.7%). Per-attack breakdown (A07–A19) + a short analysis of *why* it
resists capacity/augmentation (already partly in the KB — promote it to a documented
finding, with the anti-correlation result).

2.3 **IndexTTS-2 ceiling, stated as a result.** v9c catches held-out IndexTTS-2 ~97%, but
*fresh* BigVGAN clones sit near the front-end detectability limit. Characterise it
(score distributions, what moved it and what didn't) rather than implying it's solved.

---

## Column B — Multi-week team roadmap (real gains, not in one session)

> These are the genuinely hard, training-heavy items. They raise the *ceiling* but are
> not required to hit 9/10 — sound measurement (Column A) is. Sequenced, gated, honest.

B.1 **From-scratch SSL fine-tune for true low EER.** Per our own analysis, a head/top-
layer tweak of a robustness derivative won't break <2% on the official protocol; a
full-protocol fine-tune from the SSL backbone (RawBoost + proper aug, official train)
is the real lever. **Gate:** must beat v9c eval-EER *with CI separation*, not point gain.

B.2 **Backbone adversarial training (gated).** Prior head-only adversarial fine-tunes
failed the real-world gate (fake-detect 90→33%). A proper backbone PGD-AT run, with the
real-world + clone gates enforced, to *bound* the robustness/clean trade-off. Report the
trade honestly; ship only if gates pass. *Do not promise to "solve" PGD.*

B.3 **Broader, harder OOD benchmark.** Add the **In-the-Wild** spoof set and the
**MLAAD** slices (already on disk: `mlaad_full` 2.1 GB, `mlaad_premium` 1.1 GB) as
first-class generalisation metrics with CIs. Premium-TTS (ElevenLabs) hardening behind
the real-pass ≥90% gate.

B.4 **Diverse multi-TTS cloning corpus.** Many engines × voices × texts × channels, to
push IndexTTS-2 generalisation — a data-collection effort, explicitly scoped and
labelled "not a quick fine-tune."

---

## Suggested order of execution

1. **Phase 0** (consolidate + minDCF + entrypoint) — integrity-critical, do first.
2. **Phase 1.1–1.3** (CIs, baselines, ablations) — biggest rigor-per-hour.
3. **Phase 2** (threat model, hidden-track, IndexTTS-2 characterisation).
4. **Phase 1.4–1.5** (ruler reconciliation, repro appendix).
5. **Column B** as a separately-scoped research track with the GPU.

Completing Phases 0–2 + 1.4–1.5 is what moves the axis to **9/10**: every claim becomes
reproducible, error-barred, baseline-anchored, and honestly bounded. Column B is what
pushes toward the **research ceiling** (true <2% + bounded adversarial robustness).
