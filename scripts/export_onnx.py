#!/usr/bin/env python3
"""
Export DSFNetTiny to ONNX and apply INT8 dynamic quantization.

Target: <2MB ONNX file, <200ms latency on CPU, for edge deployment.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --checkpoint checkpoints/dsfnet_tiny/model_best.pt
    python scripts/export_onnx.py --no-quantize  # fp32 ONNX only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

OUT_DIR = Path("checkpoints/onnx")
TARGET_BYTES = 2 * 1024 * 1024  # 2MB
INPUT_SHAPE = (1, 1, 48000)     # (B, 1, T) — 3s @ 16kHz


def _load_dsfnet_tiny(ckpt_path: Path | None) -> torch.nn.Module:
    from voiceguard.models.dsfnet import DSFNetTiny
    model = DSFNetTiny()
    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model_state", state), strict=False)
        print(f"Loaded weights from {ckpt_path}")
    else:
        print("No checkpoint found — exporting with random weights (for shape validation)")
    model.eval()
    return model


class _DSFNetWrapper(torch.nn.Module):
    """Accepts pre-computed mel spectrogram so STFT never appears in the ONNX graph."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.model(waveform, spectrogram=spectrogram)


def _make_mel(model, waveform: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.mel_transform(waveform)


def export_fp32(model: torch.nn.Module, out_path: Path) -> Path:
    dummy_wav  = torch.randn(*INPUT_SHAPE)
    dummy_spec = _make_mel(model, dummy_wav)
    wrapped    = _DSFNetWrapper(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (dummy_wav, dummy_spec),
        str(out_path),
        input_names=["waveform", "spectrogram"],
        output_names=["logits"],
        dynamic_axes={"waveform": {0: "batch"}, "spectrogram": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"FP32 ONNX: {out_path}  ({size_mb:.2f} MB)")
    return out_path


def quantize_int8(fp32_path: Path, out_path: Path) -> Path:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("onnxruntime-tools not installed — skipping INT8 quantization")
        return fp32_path

    quantize_dynamic(
        str(fp32_path),
        str(out_path),
        weight_type=QuantType.QInt8,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"INT8 ONNX: {out_path}  ({size_mb:.2f} MB)  target=<2MB {'✓' if out_path.stat().st_size < TARGET_BYTES else '✗'}")
    return out_path


def benchmark(onnx_path: Path, n_runs: int = 50) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping benchmark")
        return {}

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])

    # Build feed dict matching the model's input names
    input_names = {inp.name for inp in sess.get_inputs()}
    dummy_wav = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    # Spectrogram shape — read from the first model input's actual shape
    spec_meta = [inp for inp in sess.get_inputs() if inp.name == "spectrogram"]
    if spec_meta:
        spec_shape = [d if isinstance(d, int) and d > 0 else 1 for d in spec_meta[0].shape]
        spec_shape[0] = 1  # batch=1
        dummy_spec = np.random.randn(*spec_shape).astype(np.float32)
    else:
        dummy_spec = np.random.randn(1, 1, 80, 301).astype(np.float32)
    feed = {"waveform": dummy_wav}
    if "spectrogram" in input_names:
        feed["spectrogram"] = dummy_spec
    dummy = dummy_wav  # kept for unused-var silence

    # Warmup
    for _ in range(5):
        sess.run(None, feed)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)

    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    print(f"Latency ({onnx_path.name}, CPU, n={n_runs}): p50={p50:.1f}ms  p95={p95:.1f}ms  target=<200ms {'✓' if p50 < 200 else '✗'}")
    return {"p50_ms": round(p50, 1), "p95_ms": round(p95, 1), "n_runs": n_runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    # Auto-discover checkpoint if not provided
    ckpt = args.checkpoint
    if ckpt is None:
        candidates = list(Path("checkpoints").glob("**/dsfnet*tiny*/model_best.pt"))
        ckpt = sorted(candidates)[-1] if candidates else None

    model = _load_dsfnet_tiny(ckpt)
    params = sum(p.numel() for p in model.parameters())
    print(f"DSFNetTiny params: {params:,}")

    fp32_path = args.out_dir / "dsfnet_tiny_fp32.onnx"
    export_fp32(model, fp32_path)

    results = {"fp32": {"path": str(fp32_path), "size_mb": round(fp32_path.stat().st_size / 1e6, 3)}}

    if not args.no_quantize:
        int8_path = args.out_dir / "dsfnet_tiny_int8.onnx"
        final_path = quantize_int8(fp32_path, int8_path)
        if final_path != fp32_path:
            lat = benchmark(int8_path)
            results["int8"] = {
                "path": str(int8_path),
                "size_mb": round(int8_path.stat().st_size / 1e6, 3),
                "meets_target": int8_path.stat().st_size < TARGET_BYTES,
                "latency": lat,
            }

    lat_fp32 = benchmark(fp32_path)
    results["fp32"]["latency"] = lat_fp32

    import json
    report = args.out_dir / "export_report.json"
    report.write_text(json.dumps(results, indent=2))
    print(f"\nReport: {report}")
    return results


if __name__ == "__main__":
    main()
