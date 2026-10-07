# Validation record

Run on 7 October 2026 against the original dataset.

- All three requested models trained on 155,974 records; selected XGBoost using validation weighted F1, then evaluated against 39,719 untouched test records.
- Saved and reloaded the complete pipeline successfully.
- `pytest -q` in a fresh virtual environment: **17 passed**. Six source-backed test examples match the reloaded pipeline's probabilities. Tests cover invalid/missing input, unseen context, imputation, filtering, pagination, split separation, model selection and serving the built HTML/JS/CSS.
- Vite production build passed; npm reported zero dependency vulnerabilities on installation.
- Started FastAPI and Vite; `/api/health` returned HTTP 200 with `model_loaded: true`.
- Browser submitted three held-out records through the real React form and Vite proxy:

| Record | Recorded label | Predicted label | Probability |
| --- | --- | --- | ---: |
| Nagapattinam, 2025-02-17 | Good | Satisfactory | 50.3% |
| Pudukottai, 2025-02-20 | Satisfactory | Satisfactory | 61.1% |
| Bundi, 2025-01-30 | Moderate | Moderate | 69.8% |

- Browser verified probabilities, input warnings, explanation text, and the absence of a fabricated numeric AQI.
- Browser verified model selection updates the confusion matrix and Delhi search updates the dataset table.
- Browser console inspection after these interactions: no warnings or errors.
- Phone viewport (390 × 844): dataset, prediction and guide have no horizontal page overflow; mobile menu and prediction submission work. Desktop dashboard and charts inspected visually. Viewport override reset after testing.
- Final production build is split into application and chart bundles and completes without warnings. FastAPI serves it on port 8000; a real browser prediction from this build returned successfully.
- Fresh `.venv-verify` installed `requirements.txt` without system site packages. `pip check`: no broken requirements. The suite emits two upstream deprecation warnings from the older Starlette version (multipart import and AnyIO type alias), with no test failures.
- A Vite hot-reload duplicate-root warning was found during development edits. The application component was moved to `App.jsx`, leaving root initialization in `main.jsx`. The final production build was retested successfully and showed no production console errors.
- Source SHA-256: `adb64afafded26941f52e9b12e6e108728193e607716cc203b4d428046859ea2`.
- `scripts/smoke_test.py` passed in the fresh environment: started a separate real Uvicorn server, submitted six source records over HTTP, verified probability sums, served the frontend, checked dataset filtering and invalid-input HTTP 422, then terminated its own temporary server.

The sandbox restricts local event-loop/network sockets. Local servers and the successful FastAPI test run were executed with the approved localhost access outside that restriction. This is an execution-environment constraint, not an application requirement.
