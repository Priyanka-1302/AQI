# Aero · AQI Category Classification

A complete local React + Vite + Recharts frontend, FastAPI prediction API, and reproducible scikit-learn/XGBoost training pipeline. Uses **only the existing `archive/aqi.csv`**; the source is never replaced or modified.

## Start the application

Python 3.11 and Node.js 20.19+ or 22.12+ are required. From this project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
.\scripts\start.ps1
```

Open **http://127.0.0.1:8000**. FastAPI serves both the built frontend and API. The original local workspace already has trained artifacts. **The GitHub source checkout excludes the dataset and generated model artifacts.** For a fresh clone, place the original dataset at `archive/aqi.csv` before starting; do not substitute a different dataset. If artifacts are missing, the start script trains them from that file. Stop with Ctrl+C. If local PowerShell policy prevents running scripts, first run `python -m ml.train` with the virtual environment interpreter when artifacts are missing, then use:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

To retrain (approximately 2–4 minutes on the tested machine):

```powershell
.\.venv\Scripts\python.exe -m ml.train
```

For development, run the backend above and, in a second terminal:

```powershell
cd frontend
npm run dev
```

Open **http://127.0.0.1:5173**. Vite proxies `/api` to port 8000. Ports must be free. Rebuild the frontend before using the single-server production start after frontend edits. The app is local, without authentication or a public deployment. Fonts are loaded from Google Fonts when online, with local sans-serif fallbacks; dataset and model operations do not require internet.

## Dataset and actual inputs

| Property | Value |
| --- | --- |
| Source | `archive/aqi.csv`, 29,675,276 bytes |
| Records | 235,785 |
| Date coverage | 1 April 2022 – 30 April 2025 |
| Coverage | 32 states, 291 areas |
| Original columns | 9 |
| Raw model inputs | 5 |
| Engineered features before one-hot encoding | 8 |
| Target | Existing `air_quality_status` labels |
| Categories | Good, Satisfactory, Moderate, Poor, Very Poor, Severe |

Inputs are `date`, `state`, `area`, `number_of_monitoring_stations`, and `prominent_pollutants`. Dates become four cyclic month/day-of-year components. Station count is numerical; state, area, and normalized pollutant-name combinations are one-hot encoded. **There are no measured pollutant concentrations or meteorological features.** The prediction form comes from the schema saved during training. Select options use training data; held-out examples may introduce unseen areas and trigger explicit warnings.

`aqi_value` is **excluded from inputs** because it directly encodes the category. `unit` is constant; `note` is entirely missing. Neither is predictive. No duplicates were found and no records needed exclusion. Valid high-pollution events are retained instead of being clipped as statistical outliers.

The original categories are used; no target is fabricated. One discrepancy is retained and disclosed: **Delhi, 15 November 2023, AQI 398, source label Severe**. Under the Indian standard, 398 falls in Very Poor. The unmodified source labels also contain some unusual prominent-pollutant tokens (such as SO3); the application preserves these source strings instead of inventing corrections. Original dataset authorship/provenance is not supplied with the CSV, and the app does not claim it is an official CPCB data feed.

Indian AQI ranges: Good 0–50; Satisfactory 51–100; Moderate 101–200; Poor 201–300; Very Poor 301–400; Severe 401–500. Category names and numerical values strongly indicate this scheme, apart from the disclosed discrepancy. Source for ranges and health impacts: [Government of India, National AQI announcement and tables](https://www.pib.gov.in/newsite/PrintRelease.aspx?lang=2&reg=48&relid=110654). Precautions are general summaries for the category, not individualized medical guidance.

## Training and evaluation

Exact deduplication → date/target validation → chronological split by whole date → train-only feature preprocessing and imputation → three models → validation selection → held-out test evaluation → saved pipeline.

| Partition | Dates | Records |
| --- | --- | ---: |
| Training | 2022-04-01 – 2024-05-26 | 155,974 |
| Validation | 2024-05-27 – 2024-11-11 | 40,092 |
| Test | 2024-11-12 – 2025-04-30 | 39,719 |

70%/15%/15% of unique dates are assigned chronologically. Record proportions differ because coverage changes over time. Whole dates stay within one partition. Preprocessing medians, modes, category vocabulary and sample weights use **training records only**. Numerical imputation is median, categorical imputation is mode, and unknown categories are ignored by the encoder and warned about by the API. Training weights are square-root inverse class frequency, normalized to mean one. Random state is 42.

The winner is selected by **validation weighted F1**, with macro F1 as a tie-breaker, before test metrics are computed. No hyperparameter tuning or refitting on the test data occurs. The saved model is the same training-only pipeline used for its reported test scores.

| Model | Test accuracy | Weighted precision | Weighted recall | Weighted F1 | Macro F1 | Validation weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 51.61% | 54.67% | 51.61% | 51.85% | 33.95% | 54.70% |
| Decision Tree | 49.59% | 53.58% | 49.59% | 49.02% | 32.82% | 52.75% |
| **XGBoost** | **53.87%** | **57.05%** | **53.87%** | **54.78%** | **35.51%** | **56.81%** |

All classification reports, confusion matrices, split class counts, settings, source SHA-256 and global permutation importance are in `models/model_info.json`. Model hyperparameters are explicit in `ml/train.py`. The full selected pipeline, including custom feature engineering and fitted preprocessing, is in `models/best_pipeline.joblib` (~2.4 MB).

Global importance measures validation weighted F1 decrease after shuffling each raw input (1,800 validation records, 3 repeats). The local explanation substitutes each raw input with its training median/mode and measures the change in the predicted category's probability. It is **sensitivity**, not a SHAP value or causal attribution; substitutions can form unrealistic combinations of correlated location inputs.

## Limitations

- This is **contextual classification**, not concentration-based AQI calculation, live monitoring, or a future forecast. Prominent pollutants are contemporaneous observations.
- Numeric AQI is deliberately returned as `null`. Displayed example AQI is an observed source value, never a model output.
- Probabilities are uncalibrated. Weighted F1 of 54.78% and macro F1 of 35.51% demonstrate substantial uncertainty.
- Severe has only 4 validation examples and 47 test examples. Selected-model Severe recall is **6.38%**, precision **0.80%**, and F1 **1.42%**. Do not use this classifier to make health or safety decisions.
- Seasonal and geographic coverage shifts affect the chronological test scores; performance is not a claim of accuracy for unseen future data.
- New areas and station counts outside the training range produce warnings. The API requires every input even though the saved preprocessing can impute missing training values.
- Only load trusted joblib artifacts. The model is generated locally by this project's training code and is not committed to GitHub.

## API

- `GET /api/health`: model readiness.
- `GET /api/model-info`: schema, model scores, dataset analysis, example inputs, split counts and limitations.
- `POST /api/predict`: category, probabilities, sensitivity explanation, health summary and warnings.
- `GET /api/dataset?page=1&page_size=15&search=Delhi&category=Poor`: filtered, paginated original observations.
- `GET /docs`: interactive FastAPI documentation.

Real held-out request example:

```json
{
  "date": "2025-01-30",
  "state": "Rajasthan",
  "area": "Bundi",
  "number_of_monitoring_stations": 1,
  "prominent_pollutants": "PM10"
}
```

Its recorded category is Moderate (AQI 123); the saved model predicts Moderate with ~69.8% probability. Results are not guaranteed to match every original label. Omitted, extra, malformed, empty or out-of-bounds fields return HTTP 422. Unknown categorical values return a prediction with an explicit warning.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm run build
```

The API suite checks source-backed examples from all six categories against the reloaded artifact, chronological separation, winner selection, probability sums, missing-value preprocessing, invalid requests, unseen categories and filtered pagination. Browser validation details and clean-start results are recorded in `VALIDATION.md`.

## Project layout

```text
archive/aqi.csv              original, untouched dataset
ml/pipeline.py              serializable preprocessing and model factory
ml/train.py                 training, comparison, analysis and artifact export
models/best_pipeline.joblib selected fitted pipeline
models/model_info.json      real metrics, schema and data profile
backend/main.py             API, input validation and built frontend serving
frontend/src/main.jsx      React entry point
frontend/src/App.jsx       five application sections and charts
frontend/src/styles.css    responsive visual design
tests/test_app.py           API and artifact tests
scripts/start.ps1           single-server start
```
