// MediaPipe Pose wrapper
// Initialises pose estimation and exposes results via callback

export class PoseEstimator {
  constructor() {
    this.pose = null;
    this.lastResults = null;
    this.onResults = null; // set by caller
  }

  async init() {
    this.pose = new Pose({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`
    });

    this.pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    this.pose.onResults(results => {
      this.lastResults = results;
      if (this.onResults) this.onResults(results);
    });

    await this.pose.initialize();
  }

  async send(videoElement) {
    if (!this.pose) return;
    await this.pose.send({ image: videoElement });
  }

  // Draw skeleton onto canvas context
  drawSkeleton(ctx, results, w, h) {
    if (!results?.poseLandmarks) return;
    const lm = results.poseLandmarks;

    // Connections: [start, end] landmark indices
    const connections = [
      [11,12],[11,13],[13,15],[12,14],[14,16], // arms
      [11,23],[12,24],[23,24],                  // torso
      [23,25],[25,27],[24,26],[26,28],           // legs
      [15,17],[15,19],[16,18],[16,20]            // hands
    ];

    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(124,109,250,0.8)';

    // Bones
    for (const [a, b] of connections) {
      const pa = lm[a], pb = lm[b];
      if (pa.visibility < 0.3 || pb.visibility < 0.3) continue;
      ctx.beginPath();
      ctx.moveTo(pa.x * w, pa.y * h);
      ctx.lineTo(pb.x * w, pb.y * h);
      ctx.stroke();
    }

    // Joints
    for (const p of lm) {
      if (p.visibility < 0.3) continue;
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(250,109,154,0.9)';
      ctx.fill();
    }

    ctx.restore();
  }

  // Extract joint angles relevant to kendama tricks
  // Returns array of 8 angles in degrees
  extractJointAngles(landmarks) {
    if (!landmarks) return new Array(8).fill(0);

    const lm = landmarks;
    const ang = (a, b, c) => {
      const v1 = { x: lm[a].x - lm[b].x, y: lm[a].y - lm[b].y };
      const v2 = { x: lm[c].x - lm[b].x, y: lm[c].y - lm[b].y };
      const dot = v1.x * v2.x + v1.y * v2.y;
      const mag = Math.sqrt(v1.x**2 + v1.y**2) * Math.sqrt(v2.x**2 + v2.y**2);
      if (mag === 0) return 0;
      return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
    };

    return [
      ang(13, 11, 23), // L shoulder
      ang(14, 12, 24), // R shoulder
      ang(11, 13, 15), // L elbow
      ang(12, 14, 16), // R elbow
      ang(13, 15, 17), // L wrist
      ang(14, 16, 18), // R wrist
      ang(11, 23, 25), // L hip
      ang(12, 24, 26), // R hip
    ];
  }
}