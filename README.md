# 🛡️ SentinelFlag

<div align="center">

**On-device sensitive data classification · Zero network · Privacy-by-design**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: DistilBERT](https://img.shields.io/badge/Model-DistilBERT%20INT8-blue)](https://huggingface.co/distilbert-base-uncased)
[![Runtime: ONNX](https://img.shields.io/badge/Runtime-ONNX%20Runtime-orange)](https://onnxruntime.ai/)
[![Platform: React Native](https://img.shields.io/badge/Platform-React%20Native-61DAFB)](https://reactnative.dev/)
[![Val Accuracy](https://img.shields.io/badge/Val%20Accuracy-100%25-brightgreen)](./model/training_log.json)

</div>

---

## 🧠 What is SentinelFlag?

SentinelFlag is a **privacy-first, on-device NLP classifier** that detects and categorises sensitive information in text — *before* any data ever leaves the user's device.

It runs a **quantised DistilBERT model (INT8, ~65MB)** entirely via **ONNX Runtime** on Android and iOS. No internet connection is required for inference. The only output that can leave the device is a simple sensitivity label (`0–4`) — never the raw text.

> **Think of it as a privacy firewall built into your mobile app.** It lets large AI/cloud models and backend systems receive *pre-classified* data, dramatically improving their ability to handle, redact, and route sensitive payloads — without ever exposing that raw sensitive content to the network.

---

## 🤖 How SentinelFlag Helps Large Models Categorise Data

One of the most powerful use cases for SentinelFlag is as a **pre-processing layer** that feeds structured metadata to large language models (LLMs) and cloud AI systems.

### The Problem with Raw Data

When unstructured text is sent to an LLM for analysis or storage:
- The model receives **raw sensitive data** (card numbers, SSNs, GPS coords) it may not need
- Cloud systems struggle with **GDPR/CCPA compliance** when raw PII arrives unprompted
- LLMs themselves have no inherent "sensitivity aware routing" — they treat all tokens equally

### How SentinelFlag Solves This

```
User types text
      │
      ▼
┌─────────────────────────────┐
│  SentinelFlag (on-device)   │  ← runs locally, zero latency added to API calls
│  INT8 DistilBERT + ONNX RT  │
└──────────────┬──────────────┘
               │  emits label only: { label: 2, className: "financial" }
               ▼
┌─────────────────────────────┐
│  Your App / Backend / LLM   │  ← receives the label, not the raw text
│  Routes, redacts, audits    │
└─────────────────────────────┘
```

### Concrete Benefits for LLMs and Backend Systems

| Capability | Without SentinelFlag | With SentinelFlag |
|------------|---------------------|-------------------|
| **Prompt routing** | LLM sees raw card numbers, SSNs, GPS | LLM receives `"financial"` tag; routes to secure handler |
| **Redaction pipelines** | Regex-only, brittle, misses novel patterns | Semantic detection; catches paraphrased PII |
| **GDPR / CCPA compliance** | Retroactive; data has already left device | Privacy-by-design; raw text never transmitted |
| **Model training data quality** | Sensitive data pollutes training corpora | Labels enable clean filtering and annotation |
| **Audit trails** | Raw logs contain sensitive payloads | Audit logs contain sensitivity classifications only |
| **Retrieval-Augmented Generation (RAG)** | Embeddings of PII leak into vector stores | Sensitivity labels gate what enters the vector DB |

### Example: LLM-Powered App with Sensitivity-Aware Routing

```typescript
import SentinelFlag from './sdk/src';

// 1. Classify locally before sending anywhere
const result = await SentinelFlag.classify(userInput);

if (result.isSensitive) {
  // Route to a secure, privacy-compliant API endpoint
  // or redact before sending to LLM
  await secureAPI.store({ label: result.className, redactedText: '[REDACTED]' });
} else {
  // Safe to send to general LLM / cloud service
  await llmAPI.complete({ prompt: userInput });
}
```

This pattern means your LLM never needs to "understand" PII to avoid transmitting it — SentinelFlag handles that determination entirely on the device, in ~50ms.

---

## 🗂️ Sensitivity Classes

| Label | Class | Examples |
|-------|-------|---------|
| `0` | **Non-sensitive** | General messages, scheduling, weather, public info |
| `1` | **PII** | Names, emails, phone numbers, national IDs, SSNs |
| `2` | **Financial** | Card numbers, IBANs, transactions, account balances |
| `3` | **Health** | Medical conditions, medications, lab results, diagnoses |
| `4` | **Location-critical** | GPS coordinates, home address, real-time tracking |

---

## 🏗️ Technical Architecture

```
Text input (any length)
       │
       ▼
Word-based chunking (128-token windows, 10-word overlap)
       │
       ▼
DistilBERT tokenizer (on-device)
       │
       ▼
INT8 quantised DistilBERT (ONNX Runtime · CPUExecutionProvider)
       │
       ▼
Softmax over 5 classes → argmax → sensitivity label 0–4
       │
       └─ Only the label (not the text) ever leaves the device
```

### Key Design Decisions

**Why DistilBERT?**
- **66M parameters** (vs 110M for BERT-base) — 40% smaller, 60% faster inference, 97% accuracy retention
- Pre-trained on massive text corpora — strong generalisation to unseen sensitive patterns
- INT8 quantisation (via ONNX `optimize_model`) shrinks it to **~65MB**, fitting mobile OTA budgets

**Why ONNX Runtime?**
- Official React Native package: `onnxruntime-react-native`
- Cross-platform: **Android** (NNAPI delegate) + **iOS** (Core ML delegate)
- Hardware acceleration on Snapdragon NPU and Apple Neural Engine
- Deterministic, sandboxed inference — no dynamic code execution

**Why INT8 Quantisation?**
- Reduces model size by ~4× vs FP32 with minimal accuracy loss
- Faster inference on CPUs (integer ops are cheaper than float)
- Enables deployment within typical mobile app download limits

---

## 📁 Project Structure

```
sentinelflag/
├── data/
│   ├── generate_dataset.py     # Synthetic dataset generator (Faker-based, 10k+ samples)
│   └── output/                 # train.csv · val.csv · test.csv (70/15/15 split)
│
├── model/
│   ├── train.py                # Fine-tunes DistilBERT (PyTorch + Hugging Face Transformers)
│   ├── export_onnx.py          # Exports + INT8 quantises to ONNX format
│   ├── training_log.json       # Epoch-by-epoch metrics from training run
│   ├── checkpoints/            # best_model/ · final_model/ (saved HF checkpoints)
│   └── onnx/                   # sentinelflag.onnx (FP32) · sentinelflag_int8.onnx (production)
│
├── sdk/
│   ├── src/index.ts            # React Native TypeScript SDK (classify, batch, dispose)
│   └── package.json
│
├── app/
│   └── App.tsx                 # Demo app (live classify · benchmark · about screen)
│
├── SentinelFlagDemo/           # Expo project (ready to run on device)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## ⚙️ Technologies Used

### Machine Learning Pipeline

| Technology | Role |
|-----------|------|
| **Python 3.10+** | Training pipeline language |
| **PyTorch ≥ 2.0** | Model training, gradient computation, mixed-precision (AMP) |
| **Hugging Face Transformers ≥ 4.35** | DistilBERT model + tokenizer + training utilities |
| **Hugging Face Optimum ≥ 1.13** | ONNX export + INT8 static/dynamic quantisation |
| **ONNX ≥ 1.14** | Open Neural Network Exchange format for cross-platform deployment |
| **scikit-learn** | Accuracy, F1, confusion matrix evaluation metrics |
| **Faker** | Realistic synthetic PII / financial / health / location data generation |
| **pandas / numpy** | Dataset manipulation and numerical operations |
| **AdamW + Linear LR Warmup** | Optimiser strategy (LR: 2e-5, warmup: 10%, weight decay: 0.01) |

### Mobile / App Stack

| Technology | Role |
|-----------|------|
| **React Native (Expo)** | Cross-platform mobile framework |
| **TypeScript** | SDK implementation with full type safety |
| **onnxruntime-react-native** | On-device ONNX model inference (NNAPI / Core ML delegates) |
| **Expo** | Zero-config build toolchain for iOS + Android |

### Model Details

| Parameter | Value |
|-----------|-------|
| Base model | `distilbert-base-uncased` (Hugging Face) |
| Task | 5-class sequence classification |
| Max input length | 128 tokens |
| Batch size | 32 |
| Epochs | 5 |
| Learning rate | 2e-5 |
| Quantisation | INT8 (dynamic, via ONNX Runtime) |
| Final model size | ~65MB (INT8) |

---

## 📊 Training Results

Trained on 10,000 synthetic samples (2,000 per class), 70/15/15 split:

| Epoch | Train Loss | Val Loss | Val Accuracy | Val F1 (Macro) |
|-------|-----------|---------|-------------|----------------|
| 1 | 0.5443 | 0.0091 | 100.0% | 1.0000 |
| 2 | 0.0067 | 0.0029 | 100.0% | 1.0000 |
| 3 | 0.0030 | 0.0016 | 100.0% | 1.0000 |
| 4 | 0.0020 | 0.0012 | 100.0% | 1.0000 |
| 5 | 0.0016 | 0.0011 | 100.0% | 1.0000 |

> Model achieves perfect validation F1 after epoch 1 and continues to decrease loss through epoch 5.

---

## 🚀 Setup & Run

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) CUDA GPU for faster training

### Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

For GPU training (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Step 2 — Generate training data

```bash
cd data
python generate_dataset.py
# Output: output/train.csv (7000 rows), val.csv (1500 rows), test.csv (1500 rows)
```

To generate more data (recommended for production):
```python
# Edit SAMPLES_PER_CLASS = 2000 → 10000+ in generate_dataset.py
```

### Step 3 — Train the model

```bash
cd model
python train.py
# ~15 min on GPU · ~2 hours on CPU
# Best checkpoint → checkpoints/best_model/
```

### Step 4 — Export & quantise to ONNX

```bash
python export_onnx.py
# Outputs:
#   onnx/sentinelflag.onnx       (FP32, ~250MB)
#   onnx/sentinelflag_int8.onnx  (INT8, ~65MB) ← use this on mobile
```

### Step 5 — Set up the React Native app

```bash
npx create-expo-app SentinelFlagDemo --template expo-template-blank-typescript
cd SentinelFlagDemo
npx expo install onnxruntime-react-native

# Copy the quantised model into assets
cp ../model/onnx/sentinelflag_int8.onnx ./assets/

# Copy app and SDK
cp ../app/App.tsx ./App.tsx
cp -r ../sdk ./sdk
```

### Step 6 — Run on device

```bash
# Android (requires Android Studio + connected device or emulator)
npx expo run:android

# iOS (requires Xcode on Mac)
npx expo run:ios
```

---

## 📦 SDK Usage

```typescript
import SentinelFlag from './sdk/src';

// 1. Load model once at app startup
await SentinelFlag.init();

// 2. Classify any text
const result = await SentinelFlag.classify(
  "My card number is 4539 1488 0343 6467"
);

console.log(result.label);       // 2
console.log(result.className);   // "financial"
console.log(result.confidence);  // 0.987
console.log(result.latencyMs);   // ~43ms
console.log(result.isSensitive); // true

// 3. Batch classify
const results = await SentinelFlag.classifyBatch([text1, text2, text3]);

// 4. Release model from memory when done
await SentinelFlag.dispose();
```

### `ClassificationResult` type

```typescript
interface ClassificationResult {
  label:       0 | 1 | 2 | 3 | 4;   // sensitivity class
  className:   string;                // "non-sensitive" | "PII" | "financial" | "health" | "location-critical"
  description: string;                // human-readable explanation
  confidence:  number;                // 0.0 – 1.0 softmax score for predicted class
  allScores:   number[];              // softmax scores for all 5 classes
  latencyMs:   number;                // end-to-end inference time in milliseconds
  chunkCount:  number;                // number of text chunks processed (for long inputs)
  isSensitive: boolean;               // true if label >= 1
}
```

---

## ⚡ Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Model size (INT8 ONNX) | < 200MB RAM | Fits within OTA update budgets |
| Inference latency | < 150ms | On Snapdragon 778G (mid-range Android) |
| Accuracy (test set) | > 94% macro F1 | Across all 5 sensitivity classes |
| Network calls during inference | **0** | Verified fully offline in airplane mode |

---

## 🔒 Privacy Guarantees

- ✅ **No `INTERNET` permission** required for inference
- ✅ **Raw text never transmitted** to any server or third party
- ✅ **Fully functional in airplane mode** — zero network dependency
- ✅ **Label-only output** — downstream systems cannot reverse-engineer original text from a `0–4` label
- ✅ Enables **GDPR Article 25** (Privacy by Design) compliance
- ✅ Enables **CCPA** compliance without backend architectural overhaul
- ✅ Suitable for **HIPAA-adjacent** mobile health apps where PHI must not leave the device unprotected

---

## 🗺️ Roadmap

- [ ] INT4 quantisation (target: < 35MB)
- [ ] Multilingual support via `mBERT` base
- [ ] NNAPI / Core ML hardware delegate benchmarks
- [ ] Full `vocab.txt` tokenizer loading (replace demo SimpleTokenizer)
- [ ] npm publish: `react-native-sentinelflag`
- [ ] Technical whitepaper
- [ ] Real training data augmentation (beyond Faker-generated synthetics)
- [ ] Confidence threshold UI in demo app

---

## 👤 Author

**Kiran Kumar S** — Full-Stack Developer & On-Device AI Researcher  
Kerala, India · [github.com/Kirankumars73](https://github.com/Kirankumars73)

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
