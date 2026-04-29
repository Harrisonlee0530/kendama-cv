// LSTM inference via ONNX Runtime Web
// Loads a .onnx model and runs the sliding-window trick classifier

import { TRICK_IDS, TRICK_LABELS, NUM_CLASSES, FEATURE_DIM, WINDOW_SIZE, CONF_THRESHOLD } from './labels.js';
import { buildWindowTensor } from './features.js';

export class LSTMClassifier {
  constructor() {
    this.session  = null;   // ort.InferenceSession
    this.loaded   = false;
    this._buffer  = [];     // circular feature buffer
  }

  // Load ONNX model from a File object (drag-and-drop or file input)
  async loadFromFile(file) {
    const buf = await file.arrayBuffer();
    this.session = await ort.InferenceSession.create(buf);
    this.loaded  = true;
    this._buffer = [];
    console.log('[LSTM] Model loaded. Inputs:', this.session.inputNames);
  }

  // Push one feature vector and run inference if buffer is full
  // Returns { trickId, label, confidence, scores } or null if not ready / no model
  async infer(featureVec) {
    // Always maintain buffer (used for heuristic fallback too)
    this._buffer.push(featureVec);
    if (this._buffer.length > WINDOW_SIZE) this._buffer.shift();

    if (!this.loaded || this._buffer.length < WINDOW_SIZE) return null;

    // Build input tensor [1, WINDOW_SIZE, FEATURE_DIM]
    const data = buildWindowTensor(this._buffer, WINDOW_SIZE, FEATURE_DIM);
    const tensor = new ort.Tensor('float32', data, [1, WINDOW_SIZE, FEATURE_DIM]);

    let rawScores;
    try {
      const feeds = { [this.session.inputNames[0]]: tensor };
      const results = await this.session.run(feeds);
      rawScores = results[this.session.outputNames[0]].data; // Float32Array [NUM_CLASSES]
    } catch (e) {
      console.error('[LSTM] Inference error:', e);
      return null;
    }

    const scores = softmax(Array.from(rawScores));
    const topIdx = scores.indexOf(Math.max(...scores));
    const conf   = scores[topIdx];

    if (conf < CONF_THRESHOLD) return null;

    return {
      trickId:    TRICK_IDS[topIdx],
      label:      TRICK_LABELS[topIdx],
      confidence: conf,
      scores:     TRICK_IDS.map((id, i) => ({ id, label: TRICK_LABELS[i], score: scores[i] }))
    };
  }

  // Heuristic fallback classifier — runs when no model is loaded.
  // Uses simple rule-based logic on latest tracker + pose data.
  heuristicInfer(ken, tama, jointAngles) {
    if (!ken && !tama) return null;

    const scores = new Array(NUM_CLASSES).fill(1 / NUM_CLASSES);

    if (ken && tama) {
      const kenVertical  = Math.abs(ken.angle % 90) < 25;
      const kenHoriz     = Math.abs((ken.angle % 90) - 45) < 20;
      const tamaAbove    = tama.cy < ken.cy;
      const tamaBelow    = tama.cy > ken.cy;
      const dist         = Math.sqrt((tama.cx - ken.cx)**2 + (tama.cy - ken.cy)**2);
      const close        = dist < ken.length * 0.4;
      const far          = dist > ken.length * 1.2;

      // Simple rule scores (indices match TRICK_IDS order)
      // spike, big_cup, lighthouse, around_japan, airplane, bird
      if (kenVertical && tamaAbove && close)  scores[0] += 0.5; // spike
      if (kenVertical && tamaAbove && !close) scores[1] += 0.4; // big_cup
      if (tamaBelow && close && kenVertical)  scores[2] += 0.4; // lighthouse
      if (far)                                scores[3] += 0.3; // around_japan
      if (kenHoriz && tamaAbove)              scores[4] += 0.4; // airplane
      if (kenVertical && tamaBelow)           scores[5] += 0.3; // bird
    }

    const total = scores.reduce((a, b) => a + b, 0);
    const norm  = scores.map(s => s / total);
    const topIdx = norm.indexOf(Math.max(...norm));

    return {
      trickId:    TRICK_IDS[topIdx],
      label:      TRICK_LABELS[topIdx],
      confidence: norm[topIdx],
      scores:     TRICK_IDS.map((id, i) => ({ id, label: TRICK_LABELS[i], score: norm[i] })),
      heuristic:  true
    };
  }

  clearBuffer() { this._buffer = []; }
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}