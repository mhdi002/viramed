import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import mongoose from 'mongoose';
import pino from 'pino';
import { spawn } from 'child_process';
import { randomUUID } from 'crypto';

const logger = pino({ transport: { target: 'pino-pretty' } });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const API_PREFIX = '/api';
const app = express();
app.use(cors());
app.use(express.json());

const STORAGE_DIR = path.join(__dirname, 'storage');
const MODELS_DIR = path.join(STORAGE_DIR, 'models');
const UPLOADS_DIR = path.join(STORAGE_DIR, 'uploads');
const RESULTS_DIR = path.join(STORAGE_DIR, 'results');
[STORAGE_DIR, MODELS_DIR, UPLOADS_DIR, RESULTS_DIR].forEach((d) => fs.mkdirSync(d, { recursive: true }));

const MONGO_URL = process.env.MONGO_URL;
if (!MONGO_URL) {
  logger.error('MONGO_URL not set in env');
  process.exit(1);
}
await mongoose.connect(MONGO_URL, { dbName: 'yolo_app' });

const modelSchema = new mongoose.Schema({
  id: { type: String, unique: true },
  name: { type: String, unique: true },
  filename: String,
  format: String,
  model_type: String,
  task: String,
  input_size: Number,
  classes: [String],
  created_at: { type: Date, default: () => new Date() }
}, { versionKey: false, strict: false });
const Model = mongoose.model('Model', modelSchema, 'models');

app.get(API_PREFIX + '/health', (req, res) => {
  res.json({ status: 'ok', time: new Date().toISOString() });
});

app.get(API_PREFIX + '/models', async (req, res) => {
  const models = await Model.find({}, { _id: 0 }).sort({ name: 1 }).lean();
  res.json({ models });
});

app.post(API_PREFIX + '/models/refresh', async (req, res) => {
  const supported = new Set(['.pt', '.pth', '.onnx', '.h5']);
  const files = fs.readdirSync(MODELS_DIR);
  let count = 0;
  for (const fname of files) {
    const fpath = path.join(MODELS_DIR, fname);
    if (!fs.statSync(fpath).isFile()) continue;
    const ext = path.extname(fname).toLowerCase();
    if (!supported.has(ext)) continue;
    const name = path.basename(fname, ext);
    const doc = {
      id: randomUUID(),
      name,
      filename: fname,
      format: ext.slice(1),
      model_type: 'auto',
      created_at: new Date(),
    };
    const existing = await Model.findOne({ name });
    if (existing) {
      await Model.updateOne({ name }, { $set: doc });
    } else {
      await Model.create(doc);
    }
    count++;
  }
  const models = await Model.find({}, { _id: 0 }).sort({ name: 1 }).lean();
  res.json({ count, models });
});

const upload = multer({ dest: UPLOADS_DIR });

app.post(API_PREFIX + '/models/register', upload.single('file'), async (req, res) => {
  const file = req.file;
  const { name, model_type = 'auto', task = 'detect', input_size = 640 } = req.body;
  const ext = path.extname(file.originalname).toLowerCase();
  if (!['.pt', '.pth', '.onnx', '.h5'].includes(ext)) {
    return res.status(400).json({ detail: 'Unsupported model format. Use .pt, .pth, .onnx or .h5' });
  }
  const safeName = name || path.basename(file.originalname, ext);
  const destPath = path.join(MODELS_DIR, file.originalname);
  fs.renameSync(file.path, destPath);

  const doc = {
    id: randomUUID(),
    name: safeName,
    filename: path.basename(destPath),
    format: ext.slice(1),
    model_type,
    task,
    input_size: Number(input_size),
    created_at: new Date(),
  };
  const existing = await Model.findOne({ name: safeName });
  if (existing) {
    await Model.updateOne({ name: safeName }, { $set: doc });
  } else {
    await Model.create(doc);
  }
  res.json({ model: doc });
});

function runPythonDetector(pythonScript, args, inputImageBuffer) {
  return new Promise((resolve, reject) => {
    const p = spawn('python', [pythonScript, ...args], { stdio: ['pipe', 'pipe', 'pipe'] });
    const chunks = [];
    const errs = [];
    p.stdout.on('data', (d) => chunks.push(d));
    p.stderr.on('data', (d) => errs.push(d));
    p.on('error', (e) => reject(e));
    p.on('close', (code) => {
      if (code !== 0) return reject(new Error(Buffer.concat(errs).toString() || `Python exit ${code}`));
      try {
        const out = JSON.parse(Buffer.concat(chunks).toString());
        resolve(out);
      } catch (e) { reject(e); }
    });
    if (inputImageBuffer) p.stdin.write(inputImageBuffer);
    p.stdin.end();
  });
}

app.post(API_PREFIX + '/infer', upload.single('image'), async (req, res) => {
  const { model_name, conf = 0.25, iou = 0.45, return_image = 'true', task } = req.body;
  if (!model_name) return res.status(400).json({ detail: 'model_name is required' });
  const m = await Model.findOne({ name: model_name }).lean();
  if (!m) return res.status(404).json({ detail: 'Model not found' });
  const modelPath = path.join(MODELS_DIR, m.filename);
  if (!fs.existsSync(modelPath)) return res.status(404).json({ detail: 'Model file missing on server' });

  try {
    let pyScript = path.join(__dirname, 'py', 'infer_ultralytics.py');
    let args = ['--model', modelPath, '--conf', String(conf), '--iou', String(iou), '--return_image', String(return_image)];
    let sendStdIn = true;

    if (m.format === 'onnx') {
      pyScript = path.join(__dirname, 'py', 'infer_onnx.py');
    } else if (m.format === 'pth') {
      pyScript = path.join(__dirname, 'py', 'infer_pth_adapter.py');
      sendStdIn = false; // pth adapters read from --input
      if (task || m.task) args.push('--task', String(task || m.task));
      if (req.file && req.file.path) args.push('--input', req.file.path);
      // Optional hints
      if (m.input_size) args.push('--img_size', String(m.input_size));
    } else if (m.format === 'h5') {
      pyScript = path.join(__dirname, 'py', 'infer_keras.py');
      sendStdIn = false; // keras adapter reads from --input
      if (req.file && req.file.path) args = ['--model', modelPath, '--input', req.file.path, '--return_image', String(return_image)];
    }

    const imgBuf = sendStdIn && req.file ? fs.readFileSync(req.file.path) : null;
    const out = await runPythonDetector(pyScript, args, imgBuf);
    res.json(out);
  } catch (e) {
    logger.error(e);
    res.status(500).json({ detail: e.message });
  } finally {
    if (req.file) try { fs.unlinkSync(req.file.path); } catch {}
  }
});

const PORT = process.env.PORT || 8001; // bind is externalized in deployment
app.listen(PORT, () => logger.info(`Node backend listening on ${PORT}`));