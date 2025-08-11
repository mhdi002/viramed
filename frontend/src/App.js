import React, { useEffect, useMemo, useState } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const BACKEND_URL = useMemo(() => {
    return process.env.REACT_APP_BACKEND_URL || '';
  }, []);

  const api = useMemo(() => {
    const instance = axios.create({ baseURL: BACKEND_URL });
    return instance;
  }, [BACKEND_URL]);

  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState('');
  const [image, setImage] = useState(null);
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const loadModels = async () => {
    if (!BACKEND_URL) {
      setError('REACT_APP_BACKEND_URL is missing in frontend/.env');
      return;
    }
    setError('');
    try {
      const res = await api.get('/api/models');
      setModels(res.data.models || []);
      if ((res.data.models || []).length && !selected) {
        setSelected(res.data.models[0].name);
      }
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    }
  };

  useEffect(() => {
    if (BACKEND_URL) {
      loadModels();
    }
    // eslint-disable-next-line
  }, [BACKEND_URL]);

  const refresh = async () => {
    if (!BACKEND_URL) {
      setError('REACT_APP_BACKEND_URL is missing in frontend/.env');
      return;
    }
    setError('');
    try {
      await api.post('/api/models/refresh');
      await loadModels();
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    }
  };

  const onInfer = async () => {
    if (!selected || !image) return;
    if (!BACKEND_URL) {
      setError('REACT_APP_BACKEND_URL is missing in frontend/.env');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const form = new FormData();
      form.append('model_name', selected);
      form.append('image', image);
      form.append('conf', conf);
      form.append('iou', iou);
      form.append('return_image', true);
      const res = await api.post('/api/infer', form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(res.data);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>YOLO Inference Dashboard</h1>
      </header>

      <section className="panel">
        <div className="row">
          <label>Backend URL</label>
          <code>{BACKEND_URL || '(missing REACT_APP_BACKEND_URL)'}</code>
        </div>
        <div className="row">
          <button onClick={loadModels} disabled={!BACKEND_URL}>Load Models</button>
          <button onClick={refresh} disabled={!BACKEND_URL}>Refresh Models Folder</button>
        </div>
        <div className="row">
          <label>Model</label>
          <select value={selected} onChange={e => setSelected(e.target.value)}>
            {models.map(m => (
              <option key={m.id} value={m.name}>{m.name} ({m.format})</option>
            ))}
          </select>
        </div>
        <div className="row">
          <label>Image</label>
          <input type="file" accept="image/*" onChange={e => setImage(e.target.files?.[0] || null)} />
        </div>
        <div className="row">
          <label>Conf</label>
          <input type="number" step="0.01" min="0" max="1" value={conf} onChange={e => setConf(e.target.value)} />
          <label>IoU</label>
          <input type="number" step="0.01" min="0" max="1" value={iou} onChange={e => setIou(e.target.value)} />
        </div>
        <div className="row">
          <button onClick={onInfer} disabled={loading || !image || !selected || !BACKEND_URL}>{loading ? 'Running...' : 'Run Inference'}</button>
        </div>
        {error && <div className="error">{error}</div>}
      </section>

      {result && (
        <section className="result">
          <div className="row">
            <h2>Result</h2>
          </div>
          <div className="row">
            {result.image && <img src={result.image} alt="annotated" className="annotated" />}
          </div>
          <pre className="boxes">{JSON.stringify(result.boxes, null, 2)}</pre>
        </section>
      )}
    </div>
  );
}

export default App;