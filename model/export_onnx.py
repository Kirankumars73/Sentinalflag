"""
SentinelFlag — ONNX Export + INT8 Quantisation
Compatible with: torch 2.12, transformers 5.x, onnxruntime 1.26, Python 3.13

Run: python export_onnx.py
"""

import json
import time
import numpy as np
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

BASE_DIR  = Path(__file__).parent
CKPT_DIR  = BASE_DIR / "checkpoints" / "best_model"
ONNX_DIR  = BASE_DIR / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

FP32_PATH = ONNX_DIR / "sentinelflag.onnx"
INT8_PATH = ONNX_DIR / "sentinelflag_int8.onnx"
INFO_PATH = ONNX_DIR / "model_info.json"

MAX_LEN     = 128
LABEL_NAMES = ["non-sensitive", "PII", "financial", "health", "location"]


def load():
    print(f"Loading model from: {CKPT_DIR}")
    tok   = DistilBertTokenizerFast.from_pretrained(CKPT_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(CKPT_DIR)
    model.eval()
    return model, tok


def export_fp32(model, tok):
    print("\n[1/3] Exporting FP32 ONNX (legacy exporter)...")
    dummy = tok(
        "dummy input for tracing",
        max_length=MAX_LEN, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(FP32_PATH),
            dynamo=False,                  # ← forces legacy exporter in torch 2.12
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch"},
                "attention_mask": {0: "batch"},
                "logits":         {0: "batch"},
            },
        )
    size = FP32_PATH.stat().st_size / 1e6
    print(f"  ✓ FP32 exported: {size:.1f} MB")
    return size


def quantize_int8():
    print("\n[2/3] Quantising to INT8...")

    # Fix: run in-memory shape inference before quantising
    # This corrects the shape metadata bug in opset-18 graphs
    # that causes quantize_dynamic's shape infer pass to crash
    model_fp32 = onnx.load(str(FP32_PATH))
    model_inferred = onnx.shape_inference.infer_shapes(model_fp32, check_type=False, strict_mode=False)
    inferred_path = ONNX_DIR / "sentinelflag_inferred.onnx"
    onnx.save(model_inferred, str(inferred_path))

    quantize_dynamic(
        model_input=str(inferred_path),
        model_output=str(INT8_PATH),
        weight_type=QuantType.QInt8,
    )

    inferred_path.unlink()  # clean up temp file
    size = INT8_PATH.stat().st_size / 1e6
    print(f"  ✓ INT8 saved: {size:.1f} MB")
    return size


def benchmark(path, tok, label):
    print(f"\n[3/3] Benchmarking {label}...")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    tests = [
        ("The meeting is on Tuesday at 3pm.",                               0),
        ("My email is john@example.com, phone +1-555-234-5678.",            1),
        ("Card 4539 1488 0343 6467, CVV 321, exp 09/26.",                   2),
        ("Patient has diabetes. Prescribed metformin 500mg.",               3),
        ("Home: 42 Maple Street, London SW1A 1AA.",                        4),
        ("Transfer $12,500 to IBAN GB29NWBK60161331926819.",               2),
        ("GPS: 51.507351, -0.127758. Live tracking active.",               4),
        ("Please review the attached quarterly report.",                    0),
    ]

    correct, latencies = 0, []
    for text, true_lbl in tests:
        enc = tok(text, max_length=MAX_LEN, padding="max_length",
                  truncation=True, return_tensors="np")
        feeds = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        sess.run(None, feeds)  # warm up
        t0     = time.perf_counter()
        logits = sess.run(None, feeds)[0]
        ms     = (time.perf_counter() - t0) * 1000
        pred   = int(np.argmax(logits[0]))
        ok     = pred == true_lbl
        correct += ok
        latencies.append(ms)
        print(f"  {'✓' if ok else '✗'} {LABEL_NAMES[true_lbl]:>16} → {LABEL_NAMES[pred]:<16}  {ms:.1f}ms")

    acc = correct / len(tests)
    avg = sum(latencies) / len(latencies)
    print(f"  Accuracy: {acc:.1%}  |  Avg: {avg:.1f}ms")
    return acc, avg


def main():
    print("=== SentinelFlag ONNX Export ===\n")
    model, tok = load()

    if FP32_PATH.exists():
        FP32_PATH.unlink()

    fp32_size = export_fp32(model, tok)
    int8_size = quantize_int8()

    print(f"\nSize: FP32={fp32_size:.1f}MB → INT8={int8_size:.1f}MB "
          f"({(1-int8_size/fp32_size)*100:.0f}% smaller)")

    fp32_acc, fp32_lat = benchmark(FP32_PATH, tok, "FP32")
    int8_acc, int8_lat = benchmark(INT8_PATH, tok, "INT8")

    json.dump({
        "fp32": {"size_mb": round(fp32_size,2), "accuracy": fp32_acc, "avg_ms": round(fp32_lat,2)},
        "int8": {"size_mb": round(int8_size,2), "accuracy": int8_acc, "avg_ms": round(int8_lat,2)},
        "label_names": LABEL_NAMES,
        "production_model": "sentinelflag_int8.onnx",
    }, open(INFO_PATH,"w"), indent=2)

    print(f"\n✓ Production model ready: {INT8_PATH}")
    print("  Copy sentinelflag_int8.onnx into your React Native app assets/")


if __name__ == "__main__":
    main()