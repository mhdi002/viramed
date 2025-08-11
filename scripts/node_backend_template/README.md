# Node.js YOLO/Med Imaging Backend (Template)

Mirrors the FastAPI API contract under `/api`, and delegates inference to Python scripts. Now supports:
- `.pt` Ultralytics (infer_ultralytics.py)
- `.pth` custom adapters (infer_pth_adapter.py):
  - segment3d (MS NIfTI, LightweightR2AUNet)
  - retina (2D R2UNet segmentation)
  - boneage (EffNetV2-S regression, ensemble of best_model_fold*.pth)
- `.h5` Keras classification (infer_keras.py), default labels: Benign/Malignant

## Setup

```
cd scripts/node_backend_template
# Env
echo "MONGO_URL=mongodb://localhost:27017" > .env

# Node deps
yarn

# Python deps (examples)
pip install torch timm nibabel pillow numpy ultralytics tensorflow
```

Place models under:
```
src/storage/models
```

Start:
```
yarn dev
# or
yarn start
```

## API
- POST /api/models/refresh  (now discovers .pt .pth .onnx .h5)
- POST /api/models/register  (accepts .pt .pth .onnx .h5)
- POST /api/infer
  - For `.pth` tasks, pass `task` in body (retina | segment3d | boneage)
  - For `.h5`, classification with MobileNetV2 preprocessing by default

### Examples

- Retina segmentation (.pth):
```
curl -F "model_name=retina_r2u" -F "task=retina" -F "image=@/path/img.jpg" http://localhost:8001/api/infer
```

- MS NIfTI segmentation (.pth):
```
curl -F "model_name=ms_seg" -F "task=segment3d" -F "image=@/path/volume.nii.gz" -F "return_image=true" http://localhost:8001/api/infer
```

- Bone age regression (.pth ensemble if best_model_fold*.pth found):
```
curl -F "model_name=bone_age" -F "task=boneage" -F "image=@/path/xray.jpg" http://localhost:8001/api/infer
```

- Mamography classification (.h5, default labels Benign/Malignant):
```
curl -F "model_name=mammo" -F "image=@/path/image.jpg" http://localhost:8001/api/infer
```

## Notes
- For `.pth` adapters, the request image path is passed to Python with `--input` and no stdin streaming.
- For bone age, if the registered file name contains `best_model_fold`, the adapter automatically loads all `best_model_fold*.pth` from the same directory as an ensemble.
- If your Keras `.h5` uses a different input size or labels, add `--img_size` and `--labels` to `infer_keras.py` and update server args accordingly.