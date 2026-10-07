import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from ml.pipeline import RAW_FEATURES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as client:
        yield client


def test_metadata_and_no_leakage(client):
    info = client.get('/api/model-info').json()
    assert info['dataset']['records'] == 235785
    assert info['feature_names'] == RAW_FEATURES
    assert 'aqi_value' not in info['feature_names']
    assert info['splits']['train']['end'] < info['splits']['validation']['start']
    assert info['splits']['validation']['end'] < info['splits']['test']['start']
    expected = max(info['models'], key=lambda x: (x['validation']['f1'], x['validation']['macro_f1']))['name']
    assert info['best_model'] == expected
    for model in info['models']:
        assert sum(sum(row) for row in model['test']['confusion_matrix']) == info['splits']['test']['records']


def test_built_frontend_is_served(client):
    import re
    response = client.get('/')
    assert response.status_code == 200
    assert 'Aero | AQI Category Classification' in response.text
    assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', response.text)
    assert assets
    for asset in assets:
        result = client.get(asset)
        assert result.status_code == 200
        assert result.content


def test_real_records_match_loaded_pipeline(client):
    info = client.get('/api/model-info').json()
    model = joblib.load(ROOT / 'models' / 'best_pipeline.joblib')
    source = pd.read_csv(ROOT / 'archive' / 'aqi.csv')
    assert len(info['examples']) >= 3
    for example in info['examples']:
        inputs = example['inputs']
        # Verify every example actually exists in the untouched source.
        matching = source[(source.area == inputs['area']) & (source.date == pd.Timestamp(inputs['date']).strftime('%d-%m-%Y'))]
        assert ((matching.aqi_value == example['recorded_aqi']) & (matching.air_quality_status == example['actual_category'])).any()
        response = client.post('/api/predict', json=inputs)
        assert response.status_code == 200, response.text
        result = response.json()
        expected = model.predict_proba(pd.DataFrame([inputs]))[0]
        assert result['predicted_category'] == info['categories'][int(expected.argmax())]
        assert np.allclose(list(result['probabilities'].values()), expected)
        assert abs(sum(result['probabilities'].values()) - 1) < 1e-5
        assert result['aqi_value'] is None
        assert len(result['explanation']) == len(RAW_FEATURES)


@pytest.mark.parametrize('patch', [
    {'date': '2025-02-31'}, {'date': None}, {'number_of_monitoring_stations': -1},
    {'number_of_monitoring_stations': True}, {'number_of_monitoring_stations': 1.5},
    {'number_of_monitoring_stations': 'two'}, {'state': ''}, {'state': 12},
    {'aqi_value': 155}, {'date': '01/05/2025'},
])
def test_bad_input(client, patch):
    row = client.get('/api/model-info').json()['examples'][0]['inputs']
    assert client.post('/api/predict', json={**row, **patch}).status_code == 422


def test_missing_input(client):
    assert client.post('/api/predict', json={}).status_code == 422


def test_unseen_context_warning(client):
    row = client.get('/api/model-info').json()['examples'][0]['inputs']
    result = client.post('/api/predict', json={**row, 'state': 'Unknown State', 'area': 'Unknown City'}).json()
    assert any('not seen in training' in w for w in result['warnings'])


def test_dataset_search_pagination(client):
    a = client.get('/api/dataset?search=Delhi&page_size=2').json()
    b = client.get('/api/dataset?search=Delhi&page_size=2&page=2').json()
    assert len(a['rows']) == len(b['rows']) == 2
    assert a['rows'] != b['rows']
    assert all('delhi' in (r['area'] + r['state']).lower() for r in a['rows'])
    assert client.get('/api/dataset?search=does-not-exist').json()['total'] == 0
    assert client.get('/api/dataset?page=0').status_code == 422
    assert client.get('/api/dataset?page_size=1000').status_code == 422
    assert client.get('/api/dataset?category=invalid').status_code == 422


def test_imputation_and_unknown_encoding():
    model = joblib.load(ROOT / 'models' / 'best_pipeline.joblib')
    info = json.loads((ROOT / 'models' / 'model_info.json').read_text())
    row = info['examples'][0]['inputs'].copy()
    row['number_of_monitoring_stations'] = np.nan
    row['prominent_pollutants'] = np.nan
    result = model.predict_proba(pd.DataFrame([row]))
    assert np.isfinite(result).all()
    assert abs(result.sum() - 1) < 1e-5
