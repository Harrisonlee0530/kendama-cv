// Trick class definitions and metadata

export const TRICKS = [
  {
    id: 'spike',
    label: 'Spike',
    description: 'Tama lands on ken spike',
    // Key motion signature hints (used for heuristic fallback)
    signature: { tamaAboveKen: true, kenVertical: true, fastDescent: true }
  },
  {
    id: 'big_cup',
    label: 'Big Cup',
    description: 'Tama lands in large cup',
    signature: { tamaAboveKen: true, kenVertical: false, slowDescent: true }
  },
  {
    id: 'lighthouse',
    label: 'Lighthouse',
    description: 'Ken balances on tama',
    signature: { kenAboveTama: true, kenVertical: true, lowVelocity: true }
  },
  {
    id: 'around_japan',
    label: 'Around Japan',
    description: 'Tama orbits ken full rotation',
    signature: { circularMotion: true, largeArc: true }
  },
  {
    id: 'airplane',
    label: 'Airplane',
    description: 'Tama swings to spike via hole',
    signature: { tamaAboveKen: true, kenHorizontal: true }
  },
  {
    id: 'bird',
    label: 'Bird',
    description: 'Balance trick with string taut',
    signature: { stringTaut: true, kenTilted: true, lowVelocity: true }
  }
];

export const TRICK_IDS = TRICKS.map(t => t.id);
export const TRICK_LABELS = TRICKS.map(t => t.label);
export const NUM_CLASSES = TRICKS.length;

// Feature vector dimension per frame
// 33 landmarks × 3 (x,y,vis) + 8 joint angles + 2 ken (angle, length_px) + 4 tama (relX,relY,dist,angle)
export const FEATURE_DIM = 33 * 3 + 8 + 2 + 4; // = 113

// Sliding window length (frames) fed to LSTM
export const WINDOW_SIZE = 30;

// Inference fires when confidence exceeds this threshold
export const CONF_THRESHOLD = 0.55;

// Minimum frames between logged trick events (debounce)
export const LOG_DEBOUNCE_FRAMES = 45;