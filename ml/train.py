"""Run from repository root: python -m ml.train. Never modifies the source CSV."""
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from ml.pipeline import CATEGORIES, RAW_FEATURES, make_pipeline

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'archive' / 'aqi.csv'
OUTPUT = ROOT / 'models'


def metrics(y, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average='weighted', zero_division=0)
    macro = precision_recall_fscore_support(y, predicted, average='macro', zero_division=0)[2]
    return dict(accuracy=accuracy_score(y, predicted), precision=precision, recall=recall, f1=f1, macro_f1=macro,
                confusion_matrix=confusion_matrix(y, predicted, labels=range(6)).tolist(),
                report=classification_report(y, predicted, labels=range(6), target_names=CATEGORIES, output_dict=True, zero_division=0))


def train():
    OUTPUT.mkdir(exist_ok=True)
    raw = pd.read_csv(SOURCE)
    required = set(RAW_FEATURES + ['aqi_value', 'air_quality_status'])
    if not required.issubset(raw.columns):
        raise ValueError(f'Dataset schema changed: missing {required - set(raw.columns)}')
    frame = raw.drop_duplicates().copy()
    frame['date'] = pd.to_datetime(frame['date'], format='%d-%m-%Y', errors='coerce')
    valid = frame['date'].notna() & frame['air_quality_status'].isin(CATEGORIES) & frame['aqi_value'].between(0, 500)
    excluded = int((~valid).sum())
    frame = frame.loc[valid].sort_values('date', kind='stable').reset_index(drop=True)
    expected = pd.cut(frame.aqi_value, [-1, 50, 100, 200, 300, 400, 500], labels=CATEGORIES).astype(str)
    mismatches = frame.loc[expected != frame.air_quality_status, ['date', 'state', 'area', 'aqi_value', 'air_quality_status']].copy()
    # Existing category is authoritative. Document disagreement without silently rewriting labels.
    X = frame[RAW_FEATURES].copy()
    X['date'] = X['date'].dt.strftime('%Y-%m-%d')
    y = frame.air_quality_status.map({name: i for i, name in enumerate(CATEGORIES)})
    days = np.sort(frame.date.unique())
    val_start, test_start = days[int(len(days) * .70)], days[int(len(days) * .85)]
    masks = {'train': frame.date < val_start, 'validation': (frame.date >= val_start) & (frame.date < test_start), 'test': frame.date >= test_start}
    splits = {name: {'records': int(mask.sum()), 'start': frame.loc[mask, 'date'].min().strftime('%Y-%m-%d'),
                     'end': frame.loc[mask, 'date'].max().strftime('%Y-%m-%d'),
                     'categories': frame.loc[mask, 'air_quality_status'].value_counts().reindex(CATEGORIES, fill_value=0).to_dict()}
              for name, mask in masks.items()}
    train_mask, val_mask, test_mask = masks.values()
    if set(y[train_mask]) != set(range(6)):
        raise ValueError('Training dates do not contain all six classes.')
    # Square-root inverse frequency balances minority classes without letting 0.24% dominate.
    weights = np.sqrt(compute_sample_weight('balanced', y[train_mask]))
    weights /= weights.mean()
    candidates = {
        'Random Forest': RandomForestClassifier(n_estimators=140, max_depth=24, min_samples_leaf=3, max_features=.65, n_jobs=4, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=18, min_samples_leaf=12, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=220, max_depth=7, learning_rate=.08, subsample=.85, colsample_bytree=.85,
                                 objective='multi:softprob', num_class=6, eval_metric='mlogloss', tree_method='hist', n_jobs=4, random_state=42),
    }
    fitted, scores = {}, []
    for name, estimator in candidates.items():
        start = time.time()
        print(f'Training {name} on {train_mask.sum():,} records...', flush=True)
        pipeline = make_pipeline(estimator)
        pipeline.fit(X[train_mask], y[train_mask], model__sample_weight=weights)
        validation = metrics(y[val_mask], pipeline.predict(X[val_mask]))
        fitted[name] = pipeline
        scores.append({'name': name, 'validation': validation, 'training_seconds': round(time.time() - start, 2)})
        print(f'{name}: validation weighted F1 = {validation["f1"]:.4f}', flush=True)
    best = max(scores, key=lambda item: (item['validation']['f1'], item['validation']['macro_f1']))['name']
    for result in scores:
        result['test'] = metrics(y[test_mask], fitted[result['name']].predict(X[test_mask]))
    pipeline = fitted[best]
    sample = X[val_mask].sample(n=min(1800, int(val_mask.sum())), random_state=42)
    print('Measuring permutation importance...', flush=True)
    importance = permutation_importance(pipeline, sample, y.loc[sample.index], scoring='f1_weighted', n_repeats=3, random_state=42, n_jobs=1)
    global_importance = sorted([{'feature': col, 'importance': float(importance.importances_mean[i]), 'std': float(importance.importances_std[i])}
                                for i, col in enumerate(RAW_FEATURES)], key=lambda row: row['importance'], reverse=True)
    schema = []
    baseline = {}
    training = X[train_mask]
    for col in RAW_FEATURES:
        field = {'name': col, 'label': col.replace('_', ' ').title(), 'required': True}
        if col == 'date':
            field.update(type='date', default=training[col].mode().iloc[0])
        elif pd.api.types.is_numeric_dtype(training[col]):
            field.update(type='number', min=0, max=10000, observed_min=float(training[col].min()), observed_max=float(training[col].max()), default=float(training[col].median()), step=1)
        else:
            field.update(type='select', options=sorted(training[col].dropna().unique().tolist()), default=training[col].mode().iloc[0])
        schema.append(field)
        baseline[col] = field['default']
    monthly = frame.assign(month=frame.date.dt.strftime('%Y-%m')).groupby('month').agg(mean_aqi=('aqi_value', 'mean'), records=('aqi_value', 'size')).reset_index()
    example_rows = []
    for category in CATEGORIES:
        subset = frame.loc[test_mask & (frame.air_quality_status == category)]
        if not subset.empty:
            row = subset.iloc[len(subset) // 2]
            inputs = X.loc[row.name].to_dict()
            example_rows.append({'inputs': inputs, 'actual_category': category, 'recorded_aqi': int(row.aqi_value)})
    limits = [
        'Context-only classification: the source has no pollutant concentrations or meteorological measurements.',
        'Recorded AQI is excluded because it directly encodes the target. Numeric AQI cannot be calculated from these inputs.',
        'Prominent pollutants are contemporaneous observations. This is not a future AQI forecast or a live monitoring service.',
        'Probabilities are uncalibrated model estimates, not guarantees or safety advice.',
        'Chronological validation and test periods can have different seasonal and class distributions.',
        'Rare Severe examples and unseen locations limit generalization; inspect class-wise scores.',
        'One existing target disagrees with the Indian AQI thresholds; source labels are retained and the discrepancy is disclosed.',
    ]
    numerical = raw.select_dtypes('number').dropna(axis=1, how='all')
    info = {
        'best_model': best, 'selection_metric': 'Validation weighted F1 (macro F1 breaks ties)', 'models': scores,
        'categories': CATEGORIES, 'features': schema, 'feature_names': RAW_FEATURES, 'engineered_feature_count': 8,
        'dataset': {'file': 'archive/aqi.csv', 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'raw_records': len(raw), 'records': len(frame),
                    'columns': raw.columns.tolist(), 'feature_count': len(RAW_FEATURES), 'category_count': 6,
                    'duplicates_removed': int(raw.duplicated().sum()), 'invalid_records_excluded': excluded,
                    'start': frame.date.min().strftime('%Y-%m-%d'), 'end': frame.date.max().strftime('%Y-%m-%d'),
                    'states': int(frame.state.nunique()), 'areas': int(frame.area.nunique()),
                    'missing': [{'column': c, 'count': int(raw[c].isna().sum()), 'percent': float(raw[c].isna().mean() * 100)} for c in raw],
                    'distribution': [{'name': name, 'count': int((frame.air_quality_status == name).sum())} for name in CATEGORIES],
                    'statistics': numerical.describe().round(3).to_dict(),
                    'correlation': numerical.corr().round(4).to_dict(),
                    'label_disagreements': json.loads(mismatches.to_json(orient='records', date_format='iso'))},
        'splits': splits, 'monthly': json.loads(monthly.round(2).to_json(orient='records')),
        'global_importance': global_importance, 'importance_method': 'Permutation importance: validation weighted F1 decrease; 1,800 rows, 3 repeats.',
        'examples': example_rows, 'baseline': baseline,
        'locations': {state: sorted(group.area.unique().tolist()) for state, group in training.groupby('state')},
        'limitations': limits, 'target': 'air_quality_status', 'versions': {'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__},
        'preprocessing': 'Exact duplicates removed before date split; train-only median/mode imputation and one-hot encoding; cyclic date features. Extreme valid AQI events retained. Square-root inverse-frequency training weights.',
        'excluded_features': {'aqi_value': 'Target leakage: numeric AQI defines category.', 'unit': 'Constant metadata.', 'note': 'Entirely empty.'},
        'standard_url': 'https://www.pib.gov.in/newsite/PrintRelease.aspx?lang=2&reg=48&relid=110654',
    }
    joblib.dump(pipeline, OUTPUT / 'best_pipeline.joblib', compress=3)
    (OUTPUT / 'model_info.json').write_text(json.dumps(info, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'best_model': best, 'results': [{'name': s['name'], 'test_f1': s['test']['f1'], 'test_accuracy': s['test']['accuracy']} for s in scores]}, indent=2), flush=True)


if __name__ == '__main__':
    train()
