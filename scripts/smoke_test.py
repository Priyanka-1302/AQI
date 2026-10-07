"""Start an isolated local server, test actual HTTP integration, then stop it."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8001'
process = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8001'],
    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
)
try:
    with httpx.Client(base_url=BASE, timeout=10, trust_env=False) as client:
        for _ in range(80):
            if process.poll() is not None:
                raise RuntimeError(process.stdout.read())
            try:
                if client.get('/api/health').status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(.25)
        else:
            raise RuntimeError('Server did not become ready.')
        info = client.get('/api/model-info').json()
        results = []
        for example in info['examples']:
            response = client.post('/api/predict', json=example['inputs'])
            response.raise_for_status()
            result = response.json()
            assert result['model'] == info['best_model']
            assert abs(sum(result['probabilities'].values()) - 1) < 1e-5
            results.append({'area': example['inputs']['area'], 'actual': example['actual_category'],
                            'prediction': result['predicted_category'], 'confidence': result['confidence']})
        assert client.get('/').status_code == 200
        assert client.get('/api/dataset?search=Delhi&page_size=3').json()['total'] > 0
        assert client.post('/api/predict', json={}).status_code == 422
        print(json.dumps({'status': 'passed', 'clean_server': True, 'results': results}, indent=2))
finally:
    process.terminate()
    process.wait(timeout=10)
