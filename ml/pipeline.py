"""Dataset-specific feature engineering; serialized with the fitted estimator."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORIES = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
RAW_FEATURES = ['date', 'state', 'area', 'number_of_monitoring_stations', 'prominent_pollutants']
NUMERIC = ['number_of_monitoring_stations', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
CATEGORICAL = ['state', 'area', 'prominent_pollutants']


class ContextFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        dates = pd.to_datetime(out['date'], errors='coerce')
        out['month_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
        out['month_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)
        out['day_sin'] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
        out['day_cos'] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)
        for col in CATEGORICAL:
            out[col] = out[col].apply(lambda value: value.strip() if isinstance(value, str) else np.nan)
        out['prominent_pollutants'] = out['prominent_pollutants'].apply(
            lambda value: ','.join(sorted(set(p.strip() for p in value.split(',')))) if isinstance(value, str) else np.nan)
        return out[NUMERIC + CATEGORICAL]


def make_pipeline(estimator):
    preprocess = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), NUMERIC),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ]), CATEGORICAL),
    ])
    return Pipeline([('features', ContextFeatures()), ('preprocess', preprocess), ('model', estimator)])

