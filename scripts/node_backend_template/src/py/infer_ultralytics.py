import sys, json, argparse, io
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    print(json.dumps({"error": f"Ultralytics import error: {e}"})); sys.exit(1)

def annotate(img: Image.Image, boxes):
    d = ImageDraw.Draw(img)
    for b in boxes:
        d.rectangle([(b['x1'], b['y1']), (b['x2'], b['y2'])], outline=(255,0,0), width=2)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--return_image', default='true')
    args = ap.parse_args()

    raw = sys.stdin.buffer.read()
    pil = Image.open(io.BytesIO(raw)).convert('RGB')

    model = YOLO(args.model)
    res = model.predict(source=np.array(pil), imgsz=640, conf=args.conf, iou=args.iou, verbose=False)
    boxes = []
    r0 = res[0]
    if hasattr(r0, 'boxes') and r0.boxes is not None:
        for b in r0.boxes:
            xyxy = b.xyxy[0].tolist()
            conf_v = float(b.conf[0]) if hasattr(b, 'conf') and b.conf is not None else None
            cls_v = int(b.cls[0]) if hasattr(b, 'cls') and b.cls is not None else None
            boxes.append({
                'x1': float(xyxy[0]), 'y1': float(xyxy[1]), 'x2': float(xyxy[2]), 'y2': float(xyxy[3]),
                'confidence': conf_v, 'class_id': cls_v,
            })
    img_data = None
    if str(args.return_image).lower() in ('1','true','yes','on'):
        ann = annotate(pil.copy(), boxes)
        import base64
        buf = io.BytesIO()
        ann.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        img_data = f'data:image/png;base64,{b64}'
    out = { 'boxes': boxes, 'image': img_data, 'meta': { 'count': len(boxes) } }
    print(json.dumps(out))

if __name__ == '__main__':
    main()