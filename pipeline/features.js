// Feature extraction — combines MediaPipe pose + ken/tama tracker
// into a fixed-length feature vector per frame

import { FEATURE_DIM } from './labels.js';

export class FeatureExtractor {
  constructor() {
    // Running statistics for online normalisation (Welford)
    this._n   = 0;
    this._mean = new Float32Array(FEATURE_DIM).fill(0);
    this._M2   = new Float32Array(FEATURE_DIM).fill(1);
  }

  /**
   * Build a feature vector from one frame's data.
   *
   * @param {Array}  poseLandmarks  - MediaPipe normalised landmarks (33 items)
   * @param {Array}  jointAngles    - 8 joint angles in degrees
   * @param {Array}  kenTamaVec     - 6-element vector from KenTamaTracker.featureVector()
   * @param {number} frameW         - Canvas width (for normalisation)
   * @param {number} frameH         - Canvas height
   * @returns {Float32Array} length FEATURE_DIM
   */
  extract(poseLandmarks, jointAngles, kenTamaVec, frameW, frameH) {
    const vec = new Float32Array(FEATURE_DIM);
    let idx = 0;

    // ── 1. Pose landmarks (33 × 3 = 99) ──
    if (poseLandmarks) {
      for (const lm of poseLandmarks) {
        vec[idx++] = lm.x;          // already in [0,1]
        vec[idx++] = lm.y;
        vec[idx++] = lm.visibility ?? 0;
      }
    } else {
      idx += 99;
    }

    // ── 2. Joint angles (8), normalised to [0,1] ──
    for (const a of jointAngles) {
      vec[idx++] = a / 180;
    }

    // ── 3. Ken/tama (6) ──
    for (const v of kenTamaVec) {
      vec[idx++] = v;
    }

    // Update running stats and z-score normalise
    this._updateStats(vec);
    return this._normalise(vec);
  }

  // Reset normalisation statistics (e.g. when switching subjects)
  resetStats() {
    this._n = 0;
    this._mean.fill(0);
    this._M2.fill(1);
  }

  // ── Private ──

  _updateStats(vec) {
    this._n++;
    for (let i = 0; i < FEATURE_DIM; i++) {
      const delta = vec[i] - this._mean[i];
      this._mean[i] += delta / this._n;
      const delta2 = vec[i] - this._mean[i];
      this._M2[i] += delta * delta2;
    }
  }

  _normalise(vec) {
    const out = new Float32Array(FEATURE_DIM);
    for (let i = 0; i < FEATURE_DIM; i++) {
      const std = this._n > 1 ? Math.sqrt(this._M2[i] / (this._n - 1)) : 1;
      out[i] = std > 1e-6 ? (vec[i] - this._mean[i]) / std : 0;
    }
    return out;
  }
}

// Utility: build a sliding window tensor [1, WINDOW_SIZE, FEATURE_DIM]
// from a circular buffer of feature vectors
export function buildWindowTensor(buffer, windowSize, featureDim) {
  const data = new Float32Array(windowSize * featureDim);
  const n = buffer.length;
  for (let t = 0; t < windowSize; t++) {
    // Pad with zeros if buffer not full yet
    const bufIdx = n - windowSize + t;
    if (bufIdx >= 0) {
      data.set(buffer[bufIdx], t * featureDim);
    }
  }
  return data;
}