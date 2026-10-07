import json
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import date
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ml.pipeline import CATEGORIES, RAW_FEATURES

ROOT = Path(__file__).resolve().parents[1]
HEALTH = [
    ('Minimal expected impact.', 'Continue usual activities and keep track of local air quality.'),
    ('Sensitive people may experience minor breathing discomfort.', 'Sensitive people should watch for symptoms during outdoor activity.'),
    ('People with heart or lung conditions, children and older adults may experience discomfort.', 'Vulnerable groups should reduce prolonged outdoor exertion.'),
    ('Prolonged exposure may cause breathing discomfort.', 'Reduce prolonged outdoor exertion and exposure.'),
    ('Prolonged exposure may lead to respiratory illness.', 'Limit time outdoors, especially strenuous activity.'),
    ('Respiratory effects may occur even in healthy people.', 'Avoid outdoor exertion and follow local public-health guidance.'),
]


@asynccontextmanager
async def lifespan(app):
    model_path = ROOT / 'models' / 'best_pipeline.joblib'
    info_path = ROOT / 'models' / 'model_info.json'
    if not model_path.exists() or not info_path.exists():
        raise RuntimeError('No trained artifacts. Run: python -m ml.train')
    app.state.model = joblib.load(model_path)
    app.state.info = json.loads(info_path.read_text(encoding='utf-8'))
    app.state.data = pd.read_csv(ROOT / 'archive' / 'aqi.csv').fillna('')
    yield


app = FastAPI(title='Aero · AQI Category Classification', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'], allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


@app.get('/api/health')
def health():
    return {'status': 'ok', 'model_loaded': hasattr(app.state, 'model')}


@app.get('/api/model-info')
def model_info():
    return app.state.info


@app.get('/api/dataset')
def dataset(page: int = Query(1, ge=1), page_size: int = Query(15, ge=1, le=100), search: str = Query('', max_length=100), category: str = ''):
    if category and category not in CATEGORIES:
        raise HTTPException(422, 'Unknown AQI category.')
    data = app.state.data
    if search:
        data = data.loc[data.area.str.contains(search, case=False, regex=False) | data.state.str.contains(search, case=False, regex=False)]
    if category:
        data = data.loc[data.air_quality_status == category]
    return {'total': len(data), 'page': page, 'page_size': page_size,
            'rows': data.iloc[(page - 1) * page_size:page * page_size].drop(columns=['unit', 'note']).to_dict(orient='records')}


def validate(payload):
    if set(payload) != set(RAW_FEATURES):
        raise HTTPException(422, {'message': 'Supply exactly the model input fields.', 'missing': sorted(set(RAW_FEATURES) - set(payload)), 'unexpected': sorted(set(payload) - set(RAW_FEATURES))})
    clean = {}
    warnings = []
    info = app.state.info
    for field in info['features']:
        key, value = field['name'], payload[field['name']]
        if value is None or value == '':
            raise HTTPException(422, f'{field["label"]} is required.')
        if field['type'] == 'number':
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value != int(value) or not 1 <= value <= field['max']:
                raise HTTPException(422, f'{field["label"]} must be a positive integer no greater than {field["max"]}.')
            clean[key] = int(value)
            if not field['observed_min'] <= value <= field['observed_max']:
                warnings.append(f'{field["label"]} is outside the training range.')
        elif field['type'] == 'date':
            try:
                clean[key] = date.fromisoformat(value).isoformat()
                if not 1900 <= date.fromisoformat(value).year <= 2100:
                    raise ValueError()
            except (ValueError, TypeError):
                raise HTTPException(422, 'Date must be a valid ISO date (YYYY-MM-DD), between 1900 and 2100.')
            if not info['splits']['train']['start'] <= value <= info['splits']['train']['end']:
                warnings.append('Date is outside the training period; this is a context estimate, not a forecast.')
        else:
            if not isinstance(value, str) or not value.strip() or len(value) > 120:
                raise HTTPException(422, f'{field["label"]} must be nonempty text of at most 120 characters.')
            clean[key] = value.strip()
            if clean[key] not in field['options']:
                warnings.append(f'{field["label"]} was not seen in training; confidence may be unreliable.')
    locations = info['locations']
    if clean['state'] in locations and clean['area'] not in locations[clean['state']]:
        warnings.append('This state/area combination was not seen in training.')
    return clean, warnings


@app.post('/api/predict')
def predict(payload: dict):
    inputs, warnings = validate(payload)
    model = app.state.model
    row = pd.DataFrame([inputs])
    probabilities = model.predict_proba(row)[0]
    winner = int(np.argmax(probabilities))
    category = CATEGORIES[winner]
    # Local sensitivity is a baseline substitution, not a causal/SHAP explanation.
    variations = []
    for feature in RAW_FEATURES:
        changed = inputs.copy()
        changed[feature] = app.state.info['baseline'][feature]
        variations.append(changed)
    replaced = model.predict_proba(pd.DataFrame(variations))[:, winner]
    explanation = sorted([{'feature': key, 'value': inputs[key], 'baseline': app.state.info['baseline'][key],
                           'contribution': float(probabilities[winner] - replaced[i])} for i, key in enumerate(RAW_FEATURES)], key=lambda item: abs(item['contribution']), reverse=True)
    return {'predicted_category': category, 'confidence': float(probabilities[winner]),
            'probabilities': {name: float(probabilities[i]) for i, name in enumerate(CATEGORIES)},
            'aqi_value': None, 'aqi_note': 'Numeric AQI cannot be calculated: pollutant concentrations are not in this dataset.',
            'health_interpretation': HEALTH[winner][0], 'precaution': HEALTH[winner][1],
            'explanation': explanation, 'explanation_method': 'Change in predicted-class probability when one input is replaced by its training median/mode. Correlated inputs can make substitutions unrealistic; this is sensitivity, not causality.',
            'warnings': warnings, 'model': app.state.info['best_model']}


DIST = ROOT / 'frontend' / 'dist'
if DIST.exists():
    app.mount('/', StaticFiles(directory=DIST, html=True), name='frontend')
