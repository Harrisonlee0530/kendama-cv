// pipeline/detector.js
// YOLOv8-nano object detection for ken (class 0) and tama (class 1)
// Runs via ONNX Runtime Web. Falls back to HSV tracker if no model loaded.

export const DET_INPUT_SIZE = 640; // YOLOv8 default input resolution
export const CLASS_KEN  = 0;
export const CLASS_TAMA = 1;
const CONF_THRESH = 0.35;
const IOU_THRESH  = 0.45;
const CLASSES     = ['ken', 'tama'];

export class ObjectDetector {
  constructor() {
    this.session = null;
    this.loaded  = false;

    // Off-screen canvas for frame resizing
    this._canvas = document.createElement('canvas');
    this._ctx    = this._canvas.getContext('2d', { willReadFrequently: true });

    // Last detections
    this.ken  = null; // { cx, cy, x1, y1, x2, y2, angle, length, conf }
    this.tama = null; // { cx, cy, x1, y1, x2, y2, conf }
  }

  async loadFromFile(file) {
    const buf = await file.arrayBuffer();
    this.session = await ort.InferenceSession.create(buf, {
      executionProviders: ['wasm']
    });
    this.loaded = true;
    console.log('[Detector] Model loaded. Inputs:', this.session.inputNames);
  }

  // Run detection on current video frame
  // Returns { ken, tama } — each null if not detected
  async detect(video) {
    if (!this.loaded) return { ken: null, tama: null };

    const w = video.videoWidth  || 640;
    const h = video.videoHeight || 480;

    // ── Pre-process: resize to 640×640, RGB, normalise to [0,1] ──
    this._canvas.width  = DET_INPUT_SIZE;
    this._canvas.height = DET_INPUT_SIZE;
    this._ctx.drawImage(video, 0, 0, DET_INPUT_SIZE, DET_INPUT_SIZE);
    const imgData = this._ctx.getImageData(0, 0, DET_INPUT_SIZE, DET_INPUT_SIZE).data;

    // CHW layout
    const input = new Float32Array(3 * DET_INPUT_SIZE * DET_INPUT_SIZE);
    const area  = DET_INPUT_SIZE * DET_INPUT_SIZE;
    for (let i = 0; i < area; i++) {
      input[i]          = imgData[i * 4]     / 255; // R
      input[i + area]   = imgData[i * 4 + 1] / 255; // G
      input[i + area*2] = imgData[i * 4 + 2] / 255; // B
    }

    const tensor = new ort.Tensor('float32', input,
      [1, 3, DET_INPUT_SIZE, DET_INPUT_SIZE]);

    // ── Inference ──
    let rawOutput;
    try {
      const feeds   = { [this.session.inputNames[0]]: tensor };
      const results = await this.session.run(feeds);
      rawOutput = results[this.session.outputNames[0]]; // shape [1, 6, 8400]
    } catch (e) {
      console.error('[Detector] Inference error:', e);
      return { ken: null, tama: null };
    }

    // ── Post-process ──
    const boxes = this._postprocess(rawOutput, w, h);
    this.ken  = boxes.find(b => b.cls === CLASS_KEN)  ?? null;
    this.tama = boxes.find(b => b.cls === CLASS_TAMA) ?? null;

    // Derive ken orientation from bounding box
    if (this.ken) this._computeKenOrientation(this.ken);

    return { ken: this.ken, tama: this.tama };
  }

  // Feature vector compatible with tracker.js featureVector() output
  // [kenAngle_norm, kenLength_norm, tamaRelX, tamaRelY, tamaDist_norm, tamaAngle_norm]
  featureVector(w, h) {
    const vec = new Array(6).fill(0);
    if (!this.ken) return vec;

    vec[0] = ((this.ken.angle % 180) + 180) % 180 / 180;
    vec[1] = this.ken.length / Math.sqrt(w * w + h * h);

    if (!this.tama) return vec;

    const dx = (this.tama.cx - this.ken.cx) / w;
    const dy = (this.tama.cy - this.ken.cy) / h;
    vec[2] = dx + 0.5;
    vec[3] = dy + 0.5;
    vec[4] = Math.sqrt(dx * dx + dy * dy);
    vec[5] = (Math.atan2(dy, dx) + Math.PI) / (2 * Math.PI);

    return vec;
  }

  // Draw bounding boxes + labels onto canvas context
  draw(ctx, w, h) {
    if (this.ken)  this._drawBox(ctx, this.ken,  '#facc15', 'Ken');
    if (this.tama) this._drawBox(ctx, this.tama, '#fa6d9a', 'Tama');

    // Ken orientation line
    if (this.ken) {
      ctx.save();
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth   = 2;
      const { cx, cy, angle, length } = this.ken;
      const rad = (angle * Math.PI) / 180;
      const dx  = Math.cos(rad) * length / 2;
      const dy  = Math.sin(rad) * length / 2;
      ctx.beginPath();
      ctx.moveTo(cx - dx, cy - dy);
      ctx.lineTo(cx + dx, cy + dy);
      ctx.stroke();
      ctx.restore();
    }
  }

  // ── Private ──

  // YOLOv8 output: [1, 6, 8400] → (cx,cy,w,h, cls0_conf, cls1_conf) × 8400
  _postprocess(output, origW, origH) {
    const data      = output.data;           // Float32Array
    const numDet    = output.dims[2];        // 8400
    const numFields = output.dims[1];        // 6 (4 box + 2 classes)
    const scaleX    = origW / DET_INPUT_SIZE;
    const scaleY    = origH / DET_INPUT_SIZE;

    const candidates = [];

    for (let i = 0; i < numDet; i++) {
      // column-major: field f for detection i → data[f * numDet + i]
      const bx = data[0 * numDet + i] * scaleX;
      const by = data[1 * numDet + i] * scaleY;
      const bw = data[2 * numDet + i] * scaleX;
      const bh = data[3 * numDet + i] * scaleY;

      // Find best class
      let bestCls = -1, bestConf = 0;
      for (let c = 0; c < CLASSES.length; c++) {
        const conf = data[(4 + c) * numDet + i];
        if (conf > bestConf) { bestConf = conf; bestCls = c; }
      }

      if (bestConf < CONF_THRESH) continue;

      candidates.push({
        cx: bx, cy: by,
        x1: bx - bw / 2, y1: by - bh / 2,
        x2: bx + bw / 2, y2: by + bh / 2,
        bw, bh,
        conf: bestConf,
        cls:  bestCls
      });
    }

    // NMS per class
    const kept = [];
    for (const cls of [CLASS_KEN, CLASS_TAMA]) {
      const clsDets = candidates
        .filter(d => d.cls === cls)
        .sort((a, b) => b.conf - a.conf);
      const nmsed = nms(clsDets, IOU_THRESH);
      if (nmsed.length > 0) kept.push(nmsed[0]); // top-1 per class
    }

    return kept;
  }

  // Derive ken angle and length from bounding box geometry
  _computeKenOrientation(ken) {
    // The bounding box aspect ratio tells us if the ken is more vertical or horizontal
    const dx = ken.bw;
    const dy = ken.bh;
    // angle = direction of the longer axis
    if (dy >= dx) {
      // more vertical
      ken.angle  = 90 + (dx / dy - 1) * 30; // ~90° ± tilt
      ken.length = dy;
    } else {
      // more horizontal
      ken.angle  = (dy / dx) * 30;           // ~0° ± tilt
      ken.length = dx;
    }
  }

  _drawBox(ctx, det, color, label) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);

    // Label background
    const tag = `${label} ${(det.conf * 100).toFixed(0)}%`;
    ctx.font = '12px monospace';
    const tw  = ctx.measureText(tag).width + 8;
    ctx.fillStyle = color;
    ctx.fillRect(det.x1, det.y1 - 18, tw, 18);
    ctx.fillStyle = '#000';
    ctx.fillText(tag, det.x1 + 4, det.y1 - 4);

    ctx.restore();
  }
}

// ── Non-maximum suppression ──
function iou(a, b) {
  const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
  const bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (aArea + bArea - inter + 1e-6);
}

function nms(dets, thresh) {
  const kept = [];
  const suppressed = new Set();
  for (let i = 0; i < dets.length; i++) {
    if (suppressed.has(i)) continue;
    kept.push(dets[i]);
    for (let j = i + 1; j < dets.length; j++) {
      if (iou(dets[i], dets[j]) > thresh) suppressed.add(j);
    }
  }
  return kept;
}