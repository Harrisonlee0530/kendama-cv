"""
train/train_detector.py
Fine-tune YOLOv8-nano to detect ken (class 0) and tama (class 1).
Exports to ONNX for browser inference via ONNX Runtime Web.

Usage:
    pip install ultralytics
    python train/train_detector.py --data data/detector/dataset.yaml --out models/detector.onnx

Dataset layout expected (YOLO format):
    data/detector/
        dataset.yaml
        images/
            train/   *.jpg / *.png
            val/     *.jpg / *.png
        labels/
            train/   *.txt   (one per image)
            val/     *.txt

Label format per line:
    <class_id> <cx> <cy> <w> <h>   (all normalised 0-1)
    0 = ken   1 = tama

Labelling tools: LabelImg (https://github.com/HumanSignal/labelImg)
                 or Roboflow (https://roboflow.com)
"""

import argparse
import shutil
from pathlib import Path


def train(args):
    # Import here so the script is importable without ultralytics installed
    from ultralytics import YOLO

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pretrained YOLOv8-nano ──
    # Downloads ~6 MB weights on first run
    model = YOLO('yolov8n.pt')

    # ── Fine-tune ──
    model.train(
        data      = args.data,
        epochs    = args.epochs,
        imgsz     = 640,
        batch     = args.batch,
        lr0       = args.lr,
        lrf       = 0.01,           # final LR = lr0 * lrf
        momentum  = 0.937,
        weight_decay = 5e-4,
        warmup_epochs = 3,
        cos_lr    = True,
        augment   = True,           # mosaic, flipud, fliplr, hsv jitter
        degrees   = 10,             # rotation augmentation (good for ken angle)
        scale     = 0.5,            # scale jitter
        project   = 'runs/detect',
        name      = 'kendama',
        exist_ok  = True,
        device    = args.device,
        workers   = 4,
        verbose   = True,
    )

    # ── Validate ──
    metrics = model.val()
    print(f"\nmAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")

    # ── Export to ONNX ──
    # opset 12 is broadly compatible with ONNX Runtime Web
    model.export(
        format     = 'onnx',
        imgsz      = 640,
        opset      = 12,
        simplify   = True,          # onnx-simplifier reduces graph complexity
        dynamic    = False,         # fixed batch=1 for browser
        half       = False,         # float32 required for WASM backend
    )

    # Ultralytics saves the ONNX alongside the .pt — copy to desired path
    pt_path   = Path('runs/detect/kendama/weights/best.pt')
    onnx_src  = pt_path.with_suffix('.onnx')
    if not onnx_src.exists():
        onnx_src = Path('runs/detect/kendama/weights/best.onnx')

    if onnx_src.exists():
        shutil.copy(onnx_src, args.out)
        print(f"\nDetector exported to {args.out}")
    else:
        print(f"\nWarning: could not locate exported ONNX at {onnx_src}")
        print("Check runs/detect/kendama/weights/ manually.")


# ── Dataset scaffold helper ──

def scaffold_dataset(root='data/detector'):
    """
    Create the expected directory structure and a template dataset.yaml.
    Run once before labelling images.
    """
    root = Path(root)
    for split in ['train', 'val']:
        (root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    yaml_path = root / 'dataset.yaml'
    if not yaml_path.exists():
        yaml_path.write_text(f"""\
path: {root.resolve()}
train: images/train
val:   images/val

nc: 2
names:
  0: ken
  1: tama
""")
        print(f"Created {yaml_path}")
    print(f"Dataset scaffold ready at {root}/")
    print("Add images to images/train and images/val, then label with LabelImg or Roboflow.")


# ── CLI ──

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8-nano ken/tama detector')
    sub = parser.add_subparsers(dest='cmd')

    # train subcommand
    t = sub.add_parser('train', help='Fine-tune and export detector')
    t.add_argument('--data',   default='data/detector/dataset.yaml')
    t.add_argument('--out',    default='models/detector.onnx')
    t.add_argument('--epochs', type=int,   default=50)
    t.add_argument('--batch',  type=int,   default=16)
    t.add_argument('--lr',     type=float, default=1e-3)
    t.add_argument('--device', default='',  help='cpu / 0 / 0,1 (empty = auto)')

    # scaffold subcommand
    sub.add_parser('scaffold', help='Create dataset directory structure')

    args = parser.parse_args()

    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'scaffold':
        scaffold_dataset()
    else:
        parser.print_help()