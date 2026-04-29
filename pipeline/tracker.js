// Ken + Tama tracker via HSV color segmentation
// Detects ken (elongated object) and tama (ball) from webcam frames

export class KenTamaTracker {
  constructor() {
    // HSV ranges for ken and tama, tunable by calibration
    this.kenHSV = { hLo: 10, hHi: 30, sLo: 80, sHi: 255, vLo: 60, vHi: 220 };
    this.tamaHSV = { hLo: 0, hHi: 10, sLo: 120, sHi: 255, vLo: 80, vHi: 255 };

    // Off-screen processing canvas
    this._proc = document.createElement('canvas');
    this._pctx = this._proc.getContext('2d', { willReadFrequently: true });

    // Last detected state
    this.ken = null;  // { cx, cy, angle, length }
    this.tama = null; // { cx, cy }
  }

  // Set HSV range from a hex color string (from calibration picker)
  setColorFromHex(target, hex) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    const [h, s, v] = this._rgbToHsv(r, g, b);
    const range = target === 'ken' ? this.kenHSV : this.tamaHSV;
    range.hLo = Math.max(0,   h - 15);
    range.hHi = Math.min(180, h + 15);
    range.sLo = Math.max(0,   s - 60);
    range.sHi = Math.min(255, s + 60);
    range.vLo = Math.max(0,   v - 60);
    range.vHi = Math.min(255, v + 60);
  }

  // Sample color at pixel (px, py) from video element
  sampleFromVideo(target, video, px, py) {
    this._proc.width = video.videoWidth || 640;
    this._proc.height = video.videoHeight || 480;
    this._pctx.drawImage(video, 0, 0);
    const d = this._pctx.getImageData(px, py, 1, 1).data;
    const [h, s, v] = this._rgbToHsv(d[0], d[1], d[2]);
    const range = target === 'ken' ? this.kenHSV : this.tamaHSV;
    range.hLo = Math.max(0,   h - 15);
    range.hHi = Math.min(180, h + 15);
    range.sLo = Math.max(0,   s - 50);
    range.sHi = Math.min(255, s + 50);
    range.vLo = Math.max(0,   v - 60);
    range.vHi = Math.min(255, v + 60);
  }

  // Main tracking step — call each frame with the video element
  // Returns { ken, tama } or nulls if not detected
  track(video) {
    const w = video.videoWidth  || 640;
    const h = video.videoHeight || 480;
    if (this._proc.width !== w || this._proc.height !== h) {
      this._proc.width = w; this._proc.height = h;
    }
    this._pctx.drawImage(video, 0, 0, w, h);
    const frame = this._pctx.getImageData(0, 0, w, h);
    const data = frame.data;

    // Collect masked pixels
    const kenPts = [], tamaPts = [];

    for (let i = 0; i < data.length; i += 4) {
      const [hh, ss, vv] = this._rgbToHsv(data[i], data[i+1], data[i+2]);
      const px = (i / 4) % w;
      const py = Math.floor((i / 4) / w);

      if (this._inRange(hh, ss, vv, this.kenHSV))  kenPts.push([px, py]);
      if (this._inRange(hh, ss, vv, this.tamaHSV)) tamaPts.push([px, py]);
    }

    this.ken  = kenPts.length  > 50  ? this._fitLine(kenPts,  w, h) : null;
    this.tama = tamaPts.length > 20  ? this._centroid(tamaPts, w, h) : null;

    return { ken: this.ken, tama: this.tama };
  }

  // Feature vector components from current track state (normalised 0-1)
  // Returns [kenAngle_norm, kenLength_norm, tamaRelX, tamaRelY, tamaDist_norm, tamaAngle_norm]
  featureVector(w, h) {
    const vec = new Array(6).fill(0);
    if (!this.ken) return vec;

    // Ken angle: 0=vertical, 0.5=horizontal, normalised to [0,1]
    vec[0] = ((this.ken.angle % 180) + 180) % 180 / 180;
    // Ken length normalised by image diagonal
    vec[1] = this.ken.length / Math.sqrt(w * w + h * h);

    if (!this.tama) return vec;

    // Tama position relative to ken centre
    const dx = (this.tama.cx - this.ken.cx) / w;
    const dy = (this.tama.cy - this.ken.cy) / h;
    vec[2] = dx + 0.5; // shift to [~0,1]
    vec[3] = dy + 0.5;
    vec[4] = Math.sqrt(dx*dx + dy*dy); // euclidean dist
    vec[5] = (Math.atan2(dy, dx) + Math.PI) / (2 * Math.PI); // angle [0,1]

    return vec;
  }

  // Draw tracking overlays onto canvas context
  draw(ctx, w, h) {
    if (this.ken) {
      ctx.save();
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 3;
      const { cx, cy, angle, length } = this.ken;
      const rad = (angle * Math.PI) / 180;
      const dx = Math.cos(rad) * length / 2;
      const dy = Math.sin(rad) * length / 2;
      ctx.beginPath();
      ctx.moveTo(cx - dx, cy - dy);
      ctx.lineTo(cx + dx, cy + dy);
      ctx.stroke();
      ctx.fillStyle = '#facc15';
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    if (this.tama) {
      ctx.save();
      ctx.strokeStyle = '#fa6d9a';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(this.tama.cx, this.tama.cy, 12, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }

  // ── Private helpers ──

  _inRange(h, s, v, r) {
    return h >= r.hLo && h <= r.hHi && s >= r.sLo && s <= r.sHi && v >= r.vLo && v <= r.vHi;
  }

  _centroid(pts, w, h) {
    let sx = 0, sy = 0;
    for (const [x, y] of pts) { sx += x; sy += y; }
    return { cx: sx / pts.length, cy: sy / pts.length };
  }

  // Fit a line through point cloud using PCA (covariance matrix eigenvector)
  _fitLine(pts, w, h) {
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;

    let xx = 0, xy = 0, yy = 0;
    for (const [x, y] of pts) {
      const dx = x - cx, dy = y - cy;
      xx += dx * dx; xy += dx * dy; yy += dy * dy;
    }

    // Angle of principal axis
    const angle = 0.5 * Math.atan2(2 * xy, xx - yy) * 180 / Math.PI;

    // Estimate length: project points onto principal axis
    const rad = (angle * Math.PI) / 180;
    let minP = Infinity, maxP = -Infinity;
    for (const [x, y] of pts) {
      const proj = (x - cx) * Math.cos(rad) + (y - cy) * Math.sin(rad);
      if (proj < minP) minP = proj;
      if (proj > maxP) maxP = proj;
    }
    const length = maxP - minP;

    return { cx, cy, angle, length };
  }

  _rgbToHsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const d = max - min;
    let h = 0;
    if (d !== 0) {
      if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
      else if (max === g) h = ((b - r) / d + 2) / 6;
      else h = ((r - g) / d + 4) / 6;
    }
    return [
      Math.round(h * 180),
      Math.round(max === 0 ? 0 : d / max * 255),
      Math.round(max * 255)
    ];
  }
}