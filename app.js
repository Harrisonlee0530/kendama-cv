// app.js — main orchestrator
// Wires together pose, tracker, features, LSTM and UI

import { PoseEstimator }   from './pipeline/pose.js';
import { KenTamaTracker }  from './pipeline/tracker.js';
import { ObjectDetector }  from './pipeline/detector.js';
import { FeatureExtractor } from './pipeline/features.js';
import { LSTMClassifier }  from './pipeline/lstm.js';
import { Recorder }        from './capture/recorder.js';
import { TRICK_IDS, TRICK_LABELS, NUM_CLASSES, CONF_THRESHOLD, LOG_DEBOUNCE_FRAMES } from './pipeline/labels.js';

// ── DOM refs ──
const views       = document.querySelectorAll('.view');
const navBtns     = document.querySelectorAll('.nav-btn');
const inferCanvas = document.getElementById('output-canvas');
const inferVideo  = document.getElementById('input-video');
const capCanvas   = document.getElementById('capture-canvas');
const capVideo    = document.getElementById('capture-video');
const trickName   = document.getElementById('trick-name');
const trickConf   = document.getElementById('trick-conf');
const confBars    = document.getElementById('confidence-bars');
const histList    = document.getElementById('history-list');
const modelFile    = document.getElementById('model-file');
const modelStatus  = document.getElementById('model-status');
const detectorFile = document.getElementById('detector-file');
const detectorStatus = document.getElementById('detector-status');
const kenArrow    = document.getElementById('ken-arrow');
const tamaDot     = document.getElementById('tama-dot').querySelector('::before') ?? document.getElementById('tama-dot');
const startRec    = document.getElementById('start-record');
const stopRec     = document.getElementById('stop-record');
const capStatus   = document.getElementById('capture-status');
const clipListEl  = document.getElementById('clip-list');
const exportBtn   = document.getElementById('export-dataset');
const srcRadios   = document.querySelectorAll('input[name="src"]');
const fileWrap    = document.getElementById('file-input-wrap');
const videoFileIn = document.getElementById('video-file');
const durationIn  = document.getElementById('clip-duration');
const trickSelect = document.getElementById('trick-label');
const clearHist   = document.getElementById('clear-history');

// ── Pipeline instances ──
const pose      = new PoseEstimator();
const tracker   = new KenTamaTracker();
const detector  = new ObjectDetector();
const extractor = new FeatureExtractor();
const lstm      = new LSTMClassifier();
const recorder  = new Recorder(pose, tracker, extractor, detector);

// ── State ──
let inferStream   = null;
let captureStream = null;
let animFrame     = null;
let currentView   = 'inference';
let frameCount    = 0;
let lastLogFrame  = -LOG_DEBOUNCE_FRAMES;
let samplingTarget = null; // 'ken' | 'tama' — awaiting canvas click

// ── Init ──
(async () => {
  buildConfBars();
  await pose.init();
  await startInferStream();
  setupEventListeners();
})();

// ── View switching ──
function switchView(id) {
  currentView = id;
  views.forEach(v => v.classList.toggle('active', v.id === `view-${id}`));
  navBtns.forEach(b => b.classList.toggle('active', b.dataset.view === id));
  if (id === 'inference' && !inferStream) startInferStream();
  if (id === 'capture'   && !captureStream) startCaptureStream();
}

// ── Webcam streams ──
async function startInferStream() {
  try {
    inferStream = await navigator.mediaDevices.getUserMedia({ video: true });
    inferVideo.srcObject = inferStream;
    await inferVideo.play();
    inferCanvas.width  = inferVideo.videoWidth  || 640;
    inferCanvas.height = inferVideo.videoHeight || 480;
    pose.onResults = r => renderInference(r);
    runInferLoop();
  } catch (e) {
    console.error('Camera error:', e);
    modelStatus.textContent = 'Camera access denied';
    modelStatus.className = 'error';
  }
}

async function startCaptureStream() {
  try {
    captureStream = await navigator.mediaDevices.getUserMedia({ video: true });
    capVideo.srcObject = captureStream;
    await capVideo.play();
    capCanvas.width  = capVideo.videoWidth  || 640;
    capCanvas.height = capVideo.videoHeight || 480;
    runCaptureLoop();
  } catch (e) {
    console.error('Camera error (capture):', e);
  }
}

// ── Inference loop ──
function runInferLoop() {
  const ctx = inferCanvas.getContext('2d');
  async function frame() {
    if (currentView !== 'inference') { animFrame = requestAnimationFrame(frame); return; }
    if (inferVideo.readyState >= 2) {
      inferCanvas.width  = inferVideo.videoWidth;
      inferCanvas.height = inferVideo.videoHeight;
      ctx.drawImage(inferVideo, 0, 0);
      await pose.send(inferVideo);
    }
    animFrame = requestAnimationFrame(frame);
  }
  animFrame = requestAnimationFrame(frame);
}

// Called by pose.onResults each frame
async function renderInference(results) {
  const ctx    = inferCanvas.getContext('2d');
  const w      = inferCanvas.width;
  const h      = inferCanvas.height;

  // Flip horizontal (mirror)
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(inferVideo, -w, 0, w, h);
  ctx.scale(-1, 1);
  ctx.restore();

  // Skeleton
  pose.drawSkeleton(ctx, results, w, h);

  // Ken/tama tracking
  const { ken, tama } = tracker.track(inferVideo);
  tracker.draw(ctx, w, h);

  // Update orientation indicators
  updateKenIndicator(ken);
  updateTamaIndicator(tama, ken, w, h);

  // Feature extraction
  const landmarks  = results?.poseLandmarks ?? null;
  const angles     = pose.extractJointAngles(landmarks);
  const ktVec      = tracker.featureVector(w, h);
  const vec        = extractor.extract(landmarks, angles, ktVec, w, h);

  // Run LSTM or heuristic
  let prediction = null;
  if (lstm.loaded) {
    prediction = await lstm.infer(vec);
  } else {
    prediction = lstm.heuristicInfer(ken, tama, angles);
  }

  if (prediction) updatePredictionUI(prediction);

  frameCount++;
}

// ── Capture loop ──
function runCaptureLoop() {
  const ctx = capCanvas.getContext('2d');
  async function frame() {
    if (capVideo.readyState >= 2) {
      capCanvas.width  = capVideo.videoWidth;
      capCanvas.height = capVideo.videoHeight;
      ctx.drawImage(capVideo, 0, 0);
      await pose.send(capVideo);

      const results = pose.lastResults;
      pose.drawSkeleton(ctx, results, capCanvas.width, capCanvas.height);
      tracker.track(capVideo);
      tracker.draw(ctx, capCanvas.width, capCanvas.height);

      if (recorder.isRecording) {
        const lm     = results?.poseLandmarks ?? null;
        const angles = pose.extractJointAngles(lm);
        // detector.track / tracker.track already called above in the render path
        recorder.feedFrame(lm, angles, capCanvas.width, capCanvas.height);

        // Blink border
        capCanvas.style.outline = (Math.floor(Date.now() / 500) % 2 === 0)
          ? '3px solid #fa6d9a' : 'none';
      } else {
        capCanvas.style.outline = 'none';
      }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// ── UI updates ──
function buildConfBars() {
  confBars.innerHTML = '';
  TRICK_IDS.forEach((id, i) => {
    const row = document.createElement('div');
    row.className = 'conf-row';
    row.innerHTML = `
      <div class="conf-label">
        <span>${TRICK_LABELS[i]}</span>
        <span id="score-${id}">0%</span>
      </div>
      <div class="conf-track"><div class="conf-fill" id="bar-${id}"></div></div>`;
    confBars.appendChild(row);
  });
}

function updatePredictionUI(pred) {
  trickName.textContent = pred.label;
  trickConf.textContent = (pred.confidence * 100).toFixed(1) + '%';
  trickName.style.color = pred.heuristic ? 'var(--yellow)' : 'var(--accent)';

  // Top bar highlight
  let topId = pred.trickId;
  pred.scores.forEach(({ id, score }) => {
    const bar   = document.getElementById(`bar-${id}`);
    const label = document.getElementById(`score-${id}`);
    if (!bar) return;
    bar.style.width = (score * 100).toFixed(1) + '%';
    bar.className   = 'conf-fill' + (id === topId ? ' top' : '');
    label.textContent = (score * 100).toFixed(1) + '%';
  });

  // Log to history (debounced)
  if (pred.confidence >= CONF_THRESHOLD && frameCount - lastLogFrame >= LOG_DEBOUNCE_FRAMES) {
    lastLogFrame = frameCount;
    addToHistory(pred.label);
  }
}

function addToHistory(label) {
  const li = document.createElement('li');
  const t  = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  li.innerHTML = `<span class="hl-label">${label}</span><span class="hl-time">${t}</span>`;
  histList.prepend(li);
  // Cap at 50 entries
  while (histList.children.length > 50) histList.lastChild.remove();
}

function updateKenIndicator(ken) {
  if (!ken) { kenArrow.style.opacity = '0.3'; return; }
  kenArrow.style.opacity = '1';
  kenArrow.style.transform = `rotate(${ken.angle}deg)`;
}

function updateTamaIndicator(tama, ken, w, h) {
  const dot = document.getElementById('tama-dot');
  if (!tama || !ken) { dot.style.opacity = '0.3'; return; }
  dot.style.opacity = '1';
  // Map relative position to dot pseudo-element offset within 36×36 circle
  const dx = (tama.cx - ken.cx) / w;
  const dy = (tama.cy - ken.cy) / h;
  const nx = Math.max(0.05, Math.min(0.95, dx + 0.5));
  const ny = Math.max(0.05, Math.min(0.95, dy + 0.5));
  dot.style.setProperty('--tx', (nx * 100).toFixed(1) + '%');
  dot.style.setProperty('--ty', (ny * 100).toFixed(1) + '%');
}

function renderClipList() {
  clipListEl.innerHTML = '';
  recorder.clips.forEach((clip, i) => {
    const div = document.createElement('div');
    div.className = 'clip-item';
    div.innerHTML = `
      <span class="ci-label">${clip.label}</span>
      <span class="ci-frames">${clip.frames.length} frames</span>
      <button class="ci-del" data-idx="${i}">✕</button>`;
    clipListEl.appendChild(div);
  });
}

// ── Event listeners ──
function setupEventListeners() {
  // Nav
  navBtns.forEach(b => b.addEventListener('click', () => switchView(b.dataset.view)));

  // Detector model load
  detectorFile.addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;
    detectorStatus.textContent = 'Loading…';
    detectorStatus.className = '';
    try {
      await detector.loadFromFile(file);
      detectorStatus.textContent = `✓ ${file.name}`;
      detectorStatus.className = 'loaded';
    } catch (err) {
      detectorStatus.textContent = 'Failed to load detector';
      detectorStatus.className = 'error';
      console.error(err);
    }
  });

  // Model load
  modelFile.addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;
    modelStatus.textContent = 'Loading…';
    modelStatus.className = '';
    try {
      await lstm.loadFromFile(file);
      modelStatus.textContent = `✓ ${file.name}`;
      modelStatus.className = 'loaded';
    } catch (err) {
      modelStatus.textContent = 'Failed to load model';
      modelStatus.className = 'error';
      console.error(err);
    }
  });

  // Calibration color pickers — auto-update on change
  document.getElementById('ken-color').addEventListener('input', e => {
    tracker.setColorFromHex('ken', e.target.value);
  });
  document.getElementById('tama-color').addEventListener('input', e => {
    tracker.setColorFromHex('tama', e.target.value);
  });

  // Sample buttons — next canvas click samples pixel
  document.querySelectorAll('.calib-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      samplingTarget = btn.dataset.target;
      inferCanvas.style.cursor = 'crosshair';
    });
  });
  inferCanvas.addEventListener('click', e => {
    if (!samplingTarget) return;
    const rect = inferCanvas.getBoundingClientRect();
    const scaleX = inferCanvas.width  / rect.width;
    const scaleY = inferCanvas.height / rect.height;
    const px = (e.clientX - rect.left)  * scaleX;
    const py = (e.clientY - rect.top)   * scaleY;
    tracker.sampleFromVideo(samplingTarget, inferVideo, px, py);
    samplingTarget = null;
    inferCanvas.style.cursor = 'default';
  });

  // Clear history
  clearHist.addEventListener('click', () => { histList.innerHTML = ''; });

  // Capture source toggle
  srcRadios.forEach(r => r.addEventListener('change', () => {
    fileWrap.classList.toggle('hidden', r.value !== 'file' || !r.checked);
  }));

  // Start/stop recording (webcam)
  startRec.addEventListener('click', () => {
    const label    = trickSelect.value;
    const dur      = parseInt(durationIn.value) || 0;
    const srcMode  = document.querySelector('input[name="src"]:checked').value;

    if (srcMode === 'file') {
      const file = videoFileIn.files[0];
      if (!file) { capStatus.textContent = 'Select a video file first.'; return; }
      capStatus.textContent = 'Processing video…';
      startRec.disabled = true;
      recorder.processVideoFile(file, label, p => {
        capStatus.textContent = `Processing… ${(p * 100).toFixed(0)}%`;
      }).then(clip => {
        capStatus.textContent = `Done! ${clip.frames.length} frames captured.`;
        startRec.disabled = false;
        renderClipList();
      }).catch(err => {
        capStatus.textContent = `Error: ${err.message}`;
        startRec.disabled = false;
      });
      return;
    }

    // Webcam mode
    recorder.start(label, dur);
    startRec.disabled = true;
    stopRec.disabled  = false;
    startRec.classList.add('recording');
    capStatus.textContent = dur > 0 ? `Recording ${label} for ${dur}s…` : `Recording ${label}…`;

    if (dur > 0) {
      setTimeout(() => finishRecording(), dur * 1000 + 100);
    }
  });

  stopRec.addEventListener('click', finishRecording);

  function finishRecording() {
    const clip = recorder.stop();
    startRec.disabled = false;
    stopRec.disabled  = true;
    startRec.classList.remove('recording');
    if (clip) {
      capStatus.textContent = `Saved: ${clip.label} (${clip.frames.length} frames)`;
      renderClipList();
    } else {
      capStatus.textContent = 'Clip too short, discarded.';
    }
  }

  // Delete clip
  clipListEl.addEventListener('click', e => {
    if (!e.target.classList.contains('ci-del')) return;
    recorder.removeClip(parseInt(e.target.dataset.idx));
    renderClipList();
  });

  // Export
  exportBtn.addEventListener('click', () => {
    if (recorder.clipCount === 0) { capStatus.textContent = 'No clips to export.'; return; }
    recorder.exportDataset();
  });
}