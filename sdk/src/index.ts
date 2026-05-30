/**
 * react-native-sentinelflag
 * On-device sensitive data classification — no network required.
 *
 * Usage:
 *   import SentinelFlag from 'react-native-sentinelflag';
 *
 *   await SentinelFlag.init();                        // load model once
 *   const result = await SentinelFlag.classify(text); // {label, className, confidence, latencyMs}
 *   await SentinelFlag.dispose();                     // release model from memory
 */

import { InferenceSession, Tensor } from 'onnxruntime-react-native';

// ── Types ─────────────────────────────────────────────────────────────────

export const SensitivityClass = {
  NON_SENSITIVE: 0,
  PII:           1,
  FINANCIAL:     2,
  HEALTH:        3,
  LOCATION:      4,
} as const;

export type SensitivityLabel = typeof SensitivityClass[keyof typeof SensitivityClass];

export const CLASS_NAMES: Record<SensitivityLabel, string> = {
  0: 'non-sensitive',
  1: 'PII',
  2: 'financial',
  3: 'health',
  4: 'location-critical',
};

export const CLASS_DESCRIPTIONS: Record<SensitivityLabel, string> = {
  0: 'No sensitive information detected.',
  1: 'Contains personally identifiable information (name, email, phone, ID).',
  2: 'Contains financial data (card numbers, bank accounts, transactions).',
  3: 'Contains health or medical information.',
  4: 'Contains precise location or real-time whereabouts.',
};

export interface ClassificationResult {
  label:       SensitivityLabel;
  className:   string;
  description: string;
  confidence:  number;           // 0.0 – 1.0
  allScores:   number[];         // softmax scores for all 5 classes
  latencyMs:   number;
  chunkCount:  number;           // how many chunks were processed
  isSensitive: boolean;          // true if label >= 1
}

export interface SentinelFlagConfig {
  modelPath?:   string;          // path to .onnx file (default: bundled asset)
  maxLength?:   number;          // token limit per chunk (default: 128)
  chunkOverlap?: number;         // word overlap between chunks (default: 10)
}

// ── Simple word-piece-style tokenizer ─────────────────────────────────────
// Note: this is a simplified tokenizer for demo purposes.
// In production, use the Hugging Face tokenizers WASM build or
// pre-tokenize server-side and pass token IDs directly.

const VOCAB_URL = ''; // set to your vocab.txt asset path

class SimpleTokenizer {
  private vocab: Map<string, number> = new Map();
  private readonly CLS_ID = 101;
  private readonly SEP_ID = 102;
  private readonly PAD_ID = 0;
  private readonly UNK_ID = 100;

  // Minimal working vocab for demo — replace with full vocab.txt loading
  private builtinVocab: Record<string, number> = {
    '[PAD]': 0, '[UNK]': 100, '[CLS]': 101, '[SEP]': 102, '[MASK]': 103,
    'the': 1996, 'a': 1037, 'is': 2003, 'and': 1998, 'to': 2000,
    'of': 1997, 'in': 1999, 'my': 2026, 'email': 4769, 'phone': 3042,
    'name': 2171, 'card': 4003, 'account': 4070, 'medical': 2966,
    'health': 2740, 'location': 3295, 'address': 4769, 'password': 6351,
  };

  async load(): Promise<void> {
    // In production: load full DistilBERT vocab.txt (~30,000 tokens)
    // from bundled assets and populate this.vocab
    for (const [token, id] of Object.entries(this.builtinVocid)) {
      this.vocab.set(token, id);
    }
  }

  tokenize(text: string, maxLength: number): { inputIds: number[]; attentionMask: number[] } {
    const words = text.toLowerCase().replace(/[^a-z0-9\s@.,!?-]/g, ' ').split(/\s+/).filter(Boolean);

    const tokenIds: number[] = [this.CLS_ID];
    for (const word of words) {
      tokenIds.push(this.vocab.get(word) ?? this.UNK_ID);
      if (tokenIds.length >= maxLength - 1) break;
    }
    tokenIds.push(this.SEP_ID);

    const paddingLen = maxLength - tokenIds.length;
    const inputIds      = [...tokenIds, ...Array(paddingLen).fill(this.PAD_ID)];
    const attentionMask = [...Array(tokenIds.length).fill(1), ...Array(paddingLen).fill(0)];

    return { inputIds, attentionMask };
  }

  // Chunk long text into overlapping windows
  chunkText(text: string, maxWords: number, overlapWords: number): string[] {
    const words = text.split(/\s+/);
    if (words.length <= maxWords) return [text];

    const chunks: string[] = [];
    let start = 0;
    while (start < words.length) {
      const end = Math.min(start + maxWords, words.length);
      chunks.push(words.slice(start, end).join(' '));
      start += maxWords - overlapWords;
    }
    return chunks;
  }
}

// ── Main SentinelFlag class ───────────────────────────────────────────────

class SentinelFlagClassifier {
  private session:    InferenceSession | null = null;
  private tokenizer:  SimpleTokenizer;
  private config:     Required<SentinelFlagConfig>;
  private isReady:    boolean = false;

  private readonly DEFAULT_MODEL_PATH = 'sentinelflag_int8.onnx'; // bundled asset
  private readonly DEFAULT_MAX_LENGTH  = 128;
  private readonly DEFAULT_CHUNK_OVERLAP = 10;

  constructor() {
    this.tokenizer = new SimpleTokenizer();
    this.config = {
      modelPath:    this.DEFAULT_MODEL_PATH,
      maxLength:    this.DEFAULT_MAX_LENGTH,
      chunkOverlap: this.DEFAULT_CHUNK_OVERLAP,
    };
  }

  /**
   * Load the ONNX model and tokenizer.
   * Call once at app startup (e.g. in App.tsx useEffect).
   */
  async init(config?: SentinelFlagConfig): Promise<void> {
    if (this.isReady) return;

    this.config = {
      modelPath:    config?.modelPath    ?? this.DEFAULT_MODEL_PATH,
      maxLength:    config?.maxLength    ?? this.DEFAULT_MAX_LENGTH,
      chunkOverlap: config?.chunkOverlap ?? this.DEFAULT_CHUNK_OVERLAP,
    };

    console.log('[SentinelFlag] Loading model:', this.config.modelPath);
    const t0 = Date.now();

    await this.tokenizer.load();

    this.session = await InferenceSession.create(
      this.config.modelPath,
      {
        executionProviders: ['cpu'],
        graphOptimizationLevel: 'all',
        enableCpuMemArena: true,
      }
    );

    this.isReady = true;
    console.log(`[SentinelFlag] Model ready in ${Date.now() - t0}ms`);
  }

  /**
   * Classify text for sensitivity.
   * Works fully offline — no network calls made.
   *
   * @param text  - raw text to classify (any length)
   * @returns     - ClassificationResult with label, confidence, and latency
   */
  async classify(text: string): Promise<ClassificationResult> {
    if (!this.isReady || !this.session) {
      throw new Error('[SentinelFlag] Model not loaded. Call init() first.');
    }
    if (!text || text.trim().length === 0) {
      return this.makeResult(SensitivityClass.NON_SENSITIVE, [1, 0, 0, 0, 0], 0, 1);
    }

    const t0 = performance.now();

    // Chunk long text — return highest sensitivity found across all chunks
    const maxWords = this.config.maxLength - 2; // account for [CLS] + [SEP]
    const chunks   = this.tokenizer.chunkText(text, maxWords, this.config.chunkOverlap);

    let highestLabel: SensitivityLabel = SensitivityClass.NON_SENSITIVE;
    let highestScores = [0, 0, 0, 0, 0];

    for (const chunk of chunks) {
      const { inputIds, attentionMask } = this.tokenizer.tokenize(chunk, this.config.maxLength);

      const inputIdsTensor = new Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, this.config.maxLength]);
      const maskTensor     = new Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, this.config.maxLength]);

      const outputs = await this.session.run({
        input_ids:      inputIdsTensor,
        attention_mask: maskTensor,
      });

      const logits = outputs['logits'].data as Float32Array;
      const scores = softmax(Array.from(logits));
      const label  = argmax(scores) as SensitivityLabel;

      // Keep the highest sensitivity found
      if (label > highestLabel) {
        highestLabel  = label;
        highestScores = scores;
      }
    }

    const latencyMs = performance.now() - t0;
    return this.makeResult(highestLabel, highestScores, latencyMs, chunks.length);
  }

  /**
   * Batch classify multiple texts efficiently.
   */
  async classifyBatch(texts: string[]): Promise<ClassificationResult[]> {
    return Promise.all(texts.map(t => this.classify(t)));
  }

  /**
   * Release the model from memory.
   */
  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session  = null;
      this.isReady  = false;
      console.log('[SentinelFlag] Model released.');
    }
  }

  get ready(): boolean { return this.isReady; }

  private makeResult(
    label: SensitivityLabel,
    scores: number[],
    latencyMs: number,
    chunkCount: number
  ): ClassificationResult {
    return {
      label,
      className:   CLASS_NAMES[label],
      description: CLASS_DESCRIPTIONS[label],
      confidence:  Math.round(scores[label] * 1000) / 1000,
      allScores:   scores.map(s => Math.round(s * 1000) / 1000),
      latencyMs:   Math.round(latencyMs * 10) / 10,
      chunkCount,
      isSensitive: label >= 1,
    };
  }
}

// ── Math helpers ──────────────────────────────────────────────────────────

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exp = logits.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

function argmax(arr: number[]): number {
  return arr.reduce((iMax, x, i, a) => (x > a[iMax] ? i : iMax), 0);
}

// ── Singleton export ──────────────────────────────────────────────────────
const SentinelFlag = new SentinelFlagClassifier();
export default SentinelFlag;
export { SentinelFlagClassifier };
