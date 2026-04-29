// Data collection — records labeled feature sequences from webcam or video file
// Outputs clips as JSON arrays of feature vectors ready for training

import { FEATURE_DIM } from '../pipeline/labels.js';

export class Recorder {
  constructor(poseEstimator, tracker, extractor) {
    this.pose      = poseEstimator;
    this.tracker   = tracker;
    this.extractor = extractor;

    this.isRecording = false;
    this.clips       = [];     // [{label, frames: [Float32Array, ...]}]
    this._current    = null;   // active recording
    this._timer      = null;
  }

  // Start recording a labeled clip
  // duration: seconds (0 = manual stop)
  start(label, durationSec = 0) {
    if (this.isRecording) return;
    this.isRecording = true;
    this._current = { label, frames: [], startTime: Date.now() };

    if (durationSec > 0) {
      this._timer = setTimeout(() => this.stop(), durationSec * 1000);
    }
  }

  stop() {
    if (!this.isRecording) return null;
    clearTimeout(this._timer);
    this.isRecording = false;

    const clip = this._current;
    this._current = null;

    if (clip.frames.length < 10) return null; // too short, discard

    this.clips.push(clip);
    return clip;
  }

  // Feed one frame's data — call each animation frame while recording
  feedFrame(poseLandmarks, jointAngles, kenTamaVec, w, h) {
    if (!this.isRecording || !this._current) return;
    const vec = this.extractor.extract(poseLandmarks, jointAngles, kenTamaVec, w, h);
    this._current.frames.push(Array.from(vec)); // store as plain array for JSON
  }

  // Process a video file — runs pose + tracking on every frame
  // Returns a promise that resolves when done
  async processVideoFile(file, label, onProgress) {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      video.src = URL.createObjectURL(file);
      video.muted = true;

      const procCanvas = document.createElement('canvas');
      const clip = { label, frames: [] };

      video.onloadedmetadata = () => {
        procCanvas.width  = video.videoWidth;
        procCanvas.height = video.videoHeight;
        video.currentTime = 0;
      };

      video.onseeked = async () => {
        // Send current frame to pose
        await this.pose.send(video);

        const poseResults = this.pose.lastResults;
        const landmarks   = poseResults?.poseLandmarks ?? null;
        const angles      = this.pose.extractJointAngles(landmarks);
        this.tracker.track(video);
        const ktVec = this.tracker.featureVector(video.videoWidth, video.videoHeight);

        const vec = this.extractor.extract(landmarks, angles, ktVec, video.videoWidth, video.videoHeight);
        clip.frames.push(Array.from(vec));

        const progress = video.currentTime / video.duration;
        if (onProgress) onProgress(progress);

        const nextTime = video.currentTime + (1 / 30); // sample at 30fps
        if (nextTime < video.duration) {
          video.currentTime = nextTime;
        } else {
          URL.revokeObjectURL(video.src);
          if (clip.frames.length >= 10) {
            this.clips.push(clip);
            resolve(clip);
          } else {
            reject(new Error('Video too short'));
          }
        }
      };

      video.onerror = () => reject(new Error('Failed to load video'));
      video.load();
    });
  }

  removeClip(idx) {
    this.clips.splice(idx, 1);
  }

  // Export all clips as a dataset JSON file
  exportDataset() {
    const dataset = this.clips.map(c => ({
      label:  c.label,
      frames: c.frames    // array of arrays, each length FEATURE_DIM
    }));
    const blob = new Blob([JSON.stringify(dataset, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'dataset.json';
    a.click();
    URL.revokeObjectURL(a.href);
  }

  get clipCount() { return this.clips.length; }
  get totalFrames() { return this.clips.reduce((s, c) => s + c.frames.length, 0); }
}