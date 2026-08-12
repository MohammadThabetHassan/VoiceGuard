# Evaluation Metadata

This document is the index for reproducing and reviewing the public VoiceGuard evaluation results. Every reported metric should be traceable to a checkpoint, protocol, input dataset, command, and generated artifact.

## Canonical evaluation record

Use the following fields when adding or regenerating a result:

| Field | Requirement |
|---|---|
| `checkpoint` | Exact checkpoint directory or model identifier, including the deployed lineage version. |
| `checkpoint_sha256` | SHA-256 digest of the model weights or a reference to the digest in `REPRODUCIBILITY_MANIFEST.md`. |
| `dataset` | Dataset name, release, source URL, and local path or artifact identifier. |
| `split` | Evaluation split and the train/evaluation separation assumption. |
| `protocol` | Exact scoring protocol, including clip duration, score column, and any preprocessing. |
| `command` | Command used to generate the result. |
| `output` | JSON, score cache, confidence-interval file, or canonical table produced by the command. |
| `software` | Python and key-library versions from the reproducibility manifest. |
| `limitations` | Known gaps, such as missing utterance IDs or third-party data terms. |

## Current official evaluation path

The deployed XLS-R + AASIST v9c result uses the official ASVspoof 2021 LA evaluation protocol. The canonical command is:

```bash
scripts/eval_official.sh xlsr_aasist_v9c
```

The evaluation writes the following evidence files under the configured runs directory:

```text
runs/official_xlsr_aasist_v9c.json
runs/scores_xlsr_aasist_v9c_official.npz
runs/ci_xlsr_aasist_v9c_official.json
```

The canonical table is [`RESULTS_canonical.md`](RESULTS_canonical.md). The scoring and bootstrap procedures are specified in [`EVAL_PROTOCOLS.md`](EVAL_PROTOCOLS.md), and the pinned environment and artifact hashes are recorded in [`REPRODUCIBILITY_MANIFEST.md`](REPRODUCIBILITY_MANIFEST.md).

## Dataset and split provenance

The official evaluation set is ASVspoof 2021 LA, sourced through Zenodo record `4837263` and the ASVspoof evaluation metadata. The current training/evaluation separation follows the documented protocol: the balanced training mirror is separate from the official evaluation recordings by protocol design.

That separation is a documented assumption rather than an ID-level proof when the local mirror does not retain utterance identifiers. A future release should close this gap by preserving an ID-bearing manifest and publishing the train/evaluation intersection result. Until then, this limitation must remain attached to clean-EER claims.

Clone and premium-TTS measurements use separate held-out evaluation sets. Record the synthesis engine, number of samples, speaker/text separation, and whether the engine appeared in training whenever those results are regenerated.

## Licensing and redistribution

The VoiceGuard source code and original documentation are released under the [Apache License 2.0](../LICENSE). Dataset files, pretrained checkpoints, upstream backbones, synthesis engines, fonts, images, and other third-party materials are not automatically covered by that license. Before redistributing an evaluation bundle or derived model, preserve the upstream license, attribution, and dataset access terms for every included component.

## Review checklist

Before publishing a new result:

1. Confirm the checkpoint digest and configuration file.
2. Record the dataset source, split, sample count, and protocol.
3. Run the documented evaluation command from a clean environment.
4. Preserve raw scores and confidence-interval output.
5. Regenerate the canonical results table from machine-readable output.
6. Attach known limitations and licensing notes to the result.
