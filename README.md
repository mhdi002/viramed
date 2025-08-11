# YOLO Inference – Full Project (FastAPI + React + MongoDB) with Node.js Template

A production-ready structure to run YOLO inference and visualize results.
- Backend (FastAPI) is runnable in this environment under supervisor, exposes all routes under `/api`.
- Frontend (React) consumes backend via `REACT_APP_BACKEND_URL`.
- MongoDB connection is via `MONGO_URL`.
- A separate Node.js backend template (Express) is included for your 3B requirement to drive inference via Python child-processes (Ultralytics for `.pt`, adapters for `.pth`/`.onnx`).

Important invariants (do not change):
- All backend API routes MUST be prefixed with `/api`.
- Frontend MUST read backend base URL from `REACT_APP_BACKEND_URL` (frontend/.env). Do not hardcode.
- Backend MUST read database URL from `MONGO_URL` (backend/.env). Do not hardcode.
- Backend bind stays at `0.0.0.0:8001` and frontend at port `3000` (supervisor and ingress handle mapping).

---

## Architecture Overview

- FastAPI backend
  - Routes under `/api` for health, models listing/refresh/register, and inference.
  - Storage folders auto-created at runtime: `backend/storage/{models,uploads,results}`.
  - Supported formats (FastAPI path): `.pt` (Ultralytics). `.onnx` and `.pth` return 501 here — use Node template + Python adapters.
  - Uses UUIDs instead of Mongo ObjectIDs in API responses.
- React frontend
  - Simple dashboard to call backend APIs and render annotated image.
  - Requires `REACT_APP_BACKEND_URL` to be set.
- MongoDB
  - Database name: `yolo_app`.
  - Collection: `models`.
- Node.js backend template (Express)
  - Mirrors the same API contract and shells out to Python scripts for inference.
  - Place `.pth` custom adapters and `.onnx` runtime handlers under `src/py/`.

---

## Repository Layout

```
/app
├── backend
│   ├── server.py                 # FastAPI application (prefixed /api)
│   ├── requirements.txt          # Python deps (ultralytics etc.)
│   ├── .env                      # MONGO_URL=... (already created here)
│   └── storage/
│       ├── models/               # Put .pt/.pth/.onnx files here
│       ├── uploads/
│       └── results/
├── frontend
│   ├── package.json
│   ├── .env                      # REACT_APP_BACKEND_URL=...
│   ├── public/index.html
│   └── src/
│       ├── index.js
│       ├── App.js
│       ├── App.css
│       └── index.css
└── scripts
    └── node_backend_template/    # Separate Express backend template (3B)
        ├── README.md
        ├── package.json
        └── src/
            ├── server.js         # Express server
            └── py/
                ├── infer_ultralytics.py  # .pt adapter
                ├── infer_onnx.py         # placeholder
                └── infer_pth_adapter.py  # placeholder
```

---

## Quick Start (This Environment)

This environment already runs services via supervisor.

- Check services:
  - `sudo supervisorctl status`
- Restart services (only when needed):
  - `sudo supervisorctl restart backend`
  - `sudo supervisorctl restart frontend`
  - `sudo supervisorctl restart all`
- Inspect logs (do not tail -f; print the last lines):
  - Backend: `tail -n 100 /var/log/supervisor/backend.*.log`
  - Frontend: `tail -n 100 /var/log/supervisor/frontend.*.log`

1) Place your models
- Put `.pt` / `.pth` / `.onnx` files in: `/app/backend/storage/models`.

2) Register the files into DB
- Refresh folder:
```
curl -X POST http://localhost:8001/api/models/refresh
```
- Or upload a model file:
```
curl -F "file=@/path/to/model.pt" -F "name=my_model" http://localhost:8001/api/models/register
```

3) Verify backend health and models
```
# Health
curl http://localhost:8001/api/health

# List models
curl http://localhost:8001/api/models
```

4) Configure frontend when ready
- Edit `/app/frontend/.env` and set the base URL (do not include `/api` suffix; the app adds it in paths):
```
REACT_APP_BACKEND_URL=https://your-public-backend-domain
```
- Restart frontend: `sudo supervisorctl restart frontend`
- Open the UI at the frontend URL.

Note: In this environment, Option C is active by default (frontend buttons disabled until you set the env).

---

## API Reference (FastAPI and Node template share the same contract)

Base path: `/api`

1) Health
- GET `/api/health`
- 200 Response:
```
{ "status": "ok", "time": "2025-01-01T00:00:00.000Z" }
```

2) List models
- GET `/api/models`
- 200 Response:
```
{ "models": [
  {
    "id": "uuid",
    "name": "my_model",
    "filename": "my_model.pt",
    "format": "pt|pth|onnx",
    "model_type": "auto",
    "task": "detect",
    "input_size": 640,
    "classes": ["..."],
    "created_at": "2025-01-01T00:00:00Z"
  }
]}
```

3) Refresh models folder
- POST `/api/models/refresh`
- Scans `storage/models` and upserts entries
- 200 Response: `{ "count": <int>, "models": [ ... ] }`

4) Register a model by upload
- POST `/api/models/register`
- multipart/form-data:
  - `file`: model file (.pt/.pth/.onnx)
  - `name` (optional), `model_type` (default "auto"), `task` (default "detect"), `input_size` (default 640)
- 200 Response: `{ "model": { ... } }`

5) Inference
- POST `/api/infer`
- multipart/form-data:
  - `image`: image file
  - `model_name`: name of a registered model
  - `conf` (default 0.25), `iou` (default 0.45), `return_image` (true/false)
- 200 Response:
```
{
  "boxes": [
    { "x1": 10.2, "y1": 22.9, "x2": 111.0, "y2": 200.7, "confidence": 0.93, "class_id": 0, "label": "polyp" }
  ],
  "image": "data:image/png;base64,...." ,
  "meta": { "model_name": "my_model", "format": "pt", "count": 1 }
}
```

Notes:
- FastAPI backend supports `.pt` (Ultralytics). `.onnx` and `.pth` return 501 here.
- For `.onnx` or `.pth`, use the Node.js backend template that delegates to Python adapters under `src/py/`.

---

## Frontend Usage

- The React app only calls the backend through the environment variable `REACT_APP_BACKEND_URL`.
- Never hardcode URLs in code; ensure paths keep the `/api` prefix.
- After changing `.env`, run: `sudo supervisorctl restart frontend`.

---

## Node.js Backend Template (3B – Python child-process for inference)

Location: `/app/scripts/node_backend_template`

When to use it
- If your additional models are `.pth` with custom loaders or `.onnx`, run this backend instead of the FastAPI backend, or run both depending on your infra.

Setup
```
cd /app/scripts/node_backend_template
# Create .env with Mongo URL (example)
echo "MONGO_URL=mongodb://localhost:27017" > .env

# Install Node deps
yarn

# Ensure Python env has required packages (system level)
pip install ultralytics pillow numpy

# Place models
mkdir -p src/storage/models
cp /path/to/your/models/* src/storage/models/

# Start (dev)
yarn dev
# or start
# yarn start
```

Adapters
- `.pt` uses `src/py/infer_ultralytics.py` (Ultralytics) and can return an annotated image.
- `.onnx` uses `src/py/infer_onnx.py` (placeholder) — replace with your ONNX logic (onnxruntime-node or Python onnxruntime).
- `.pth` uses `src/py/infer_pth_adapter.py` (placeholder) — place your loader & preprocessing there.

API Endpoints
- Identical to FastAPI backend (see API Reference), still under `/api`.

---

## Deployment Guide

This repository is ready to be containerized and deployed behind an ingress that routes:
- `/api` → backend service (internally on 8001)
- everything else → frontend service (port 3000)

1) Build images (example Dockerfiles)

Backend Dockerfile (example):
```
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
WORKDIR /app/backend
ENV PORT=8001 HOST=0.0.0.0
# Your process manager / entrypoint should start uvicorn binding 0.0.0.0:8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
```

Frontend Dockerfile (example with CRA):
```
FROM node:20-alpine as build
WORKDIR /app
COPY frontend/package.json yarn.lock ./
RUN yarn install --frozen-lockfile
COPY frontend/ ./
# Build uses REACT_APP_BACKEND_URL from environment at build time if needed
RUN yarn build

FROM nginx:stable-alpine
COPY --from=build /app/build /usr/share/nginx/html
# Optional: add nginx config to forward /api to backend at the ingress level
```

Node.js Template Dockerfile (example):
```
FROM node:20
WORKDIR /app
COPY scripts/node_backend_template/package.json ./
RUN yarn install --frozen-lockfile
COPY scripts/node_backend_template/ ./
ENV PORT=8001
CMD ["node", "src/server.js"]
```

2) Kubernetes manifests (high-level example)

- Ingress (pseudocode):
```
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: yolo-ingress
spec:
  rules:
  - host: your.domain
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: yolo-backend
            port:
              number: 8001
      - path: /
        pathType: Prefix
        backend:
          service:
            name: yolo-frontend
            port:
              number: 3000
```

- Backend Deployment (ensure `MONGO_URL` is set via Secret or ConfigMap)
- Frontend Deployment (ensure `REACT_APP_BACKEND_URL` is provided accordingly)

3) Env configuration
- Backend: `MONGO_URL` only (Secret/ConfigMap).
- Frontend: `REACT_APP_BACKEND_URL` only. All API requests must keep the `/api` prefix.

4) Health checks
- Backend readiness/liveness: GET `/api/health`.

5) GPU & performance (optional)
- For heavy workloads, run the Node.js template with Python adapters inside a CUDA-enabled image and ensure Ultralytics/torch with CUDA.
- Batch inference or queueing can be added behind `/api/infer`.

---

## Troubleshooting

- 404 from frontend calls:
  - Likely `REACT_APP_BACKEND_URL` not set or missing `/api` prefix on route paths. Set the env and restart frontend.
- `MONGO_URL not set` at backend startup:
  - Ensure `/app/backend/.env` contains `MONGO_URL=...`.
- `Model not found` on inference:
  - Run `/api/models/refresh` or `/api/models/register`, ensure filename exists in `storage/models`.
- ONNX or PTH not supported in FastAPI path:
  - Use Node.js template and implement the adapters in `src/py/`.
- File watch limit reached in logs:
  - Happens with hot reload watchers; it’s transient under supervisor. Avoid massive file churn in backend folder.

---

## Data Contracts

- All IDs are UUID strings.
- Boxes are returned as `{ x1, y1, x2, y2, confidence?, class_id?, label? }`.
- Annotated image is a `data:image/png;base64,...` data URL if `return_image=true`.

---

## Versions (selected)

- Python FastAPI stack
  - fastapi 0.111.x, starlette 0.37.x, pydantic 2.7.x
  - ultralytics 8.3.x
  - pillow, numpy, pymongo
- Frontend
  - react 18, react-scripts 5
- Node.js template
  - express 4, mongoose 8, multer, pino, nodemon (dev)

---

## Example cURL Snippets

Refresh folder:
```
curl -X POST "$REACT_APP_BACKEND_URL/api/models/refresh"
```

Register a model by upload:
```
curl -F "file=@/models/polyp.pt" -F "name=polyp" "$REACT_APP_BACKEND_URL/api/models/register"
```

Inference (.pt through FastAPI):
```
curl -F "model_name=polyp" -F "image=@/images/sample.jpg" -F "conf=0.25" -F "iou=0.45" -F "return_image=true" \
  "$REACT_APP_BACKEND_URL/api/infer"
```

---

## Maintenance Commands (this environment)

- Restart services via supervisor:
```
sudo supervisorctl restart backend
sudo supervisorctl restart frontend
sudo supervisorctl restart all
```

- View last backend logs lines:
```
tail -n 100 /var/log/supervisor/backend.*.log
```

- View last frontend logs lines:
```
tail -n 100 /var/log/supervisor/frontend.*.log
```

---

## Notes
- Never hardcode URLs or ports; always use the environment variables described above.
- Keep `/api` prefix on all backend routes to match ingress.
- For `.pth` and `.onnx` use cases, prefer the Node.js template with Python adapters.

Happy building! 🚀