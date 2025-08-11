import sys, json, argparse
import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except Exception as e:
    print(json.dumps({"error": f"TensorFlow/Keras import error: {e}"}))
    sys.exit(1)

DEFAULT_LABELS = ["Benign", "Malignant"]

def predict_keras(model_path: str, image_path: str, img_size: int = 224, labels=None):
    labels = labels or DEFAULT_LABELS
    model = load_model(model_path)
    img = load_img(image_path, target_size=(img_size, img_size))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = model.predict(arr)
    if preds.ndim == 2 and preds.shape[1] == 1:
        # binary sigmoid -> convert to [p0, p1]
        p1 = float(preds[0][0])
        p0 = 1.0 - p1
        probs = [p0, p1]
    else:
        # assume softmax
        probs = preds[0].tolist()
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return {
        "boxes": [],
        "image": None,
        "meta": {
            "task": "classify",
            "class": labels[idx] if 0 <= idx < len(labels) else str(idx),
            "confidence": conf,
            "probs": probs,
            "labels": labels,
        }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--labels', default=','.join(DEFAULT_LABELS))
    ap.add_argument('--return_image', default='false')
    args = ap.parse_args()
    labels = [s.strip() for s in args.labels.split(',') if s.strip()]
    try:
        out = predict_keras(args.model, args.input, args.img_size, labels)
        print(json.dumps(out))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()