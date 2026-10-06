# Student Placement & Salary Predictor

A full-stack machine-learning application that estimates a student's placement probability and, if placed, a likely salary and salary interval. The project includes the dataset and analysis notebook, trained model artifacts, a Flask API, and a schema-driven React + Vite frontend.

> **Use responsibly:** This is an educational demonstration, not a hiring or admissions tool. Placement outcomes depend on many factors that the dataset cannot represent. The placement estimate is less reliable than the salary estimate; see [model performance and limitations](#model-performance-and-limitations).

## Contents

- [Student Placement \& Salary Predictor](#student-placement--salary-predictor)
  - [Contents](#contents)
  - [Project at a glance](#project-at-a-glance)
  - [Repository structure](#repository-structure)
  - [Run locally](#run-locally)
    - [Prerequisites](#prerequisites)
    - [1. Start the backend (Windows PowerShell)](#1-start-the-backend-windows-powershell)
    - [2. Start the frontend](#2-start-the-frontend)
  - [How predictions work](#how-predictions-work)
  - [Input features](#input-features)
  - [Model performance and limitations](#model-performance-and-limitations)
  - [Backend API](#backend-api)
  - [Frontend](#frontend)
  - [ML artifacts and reports](#ml-artifacts-and-reports)
    - [`ML/01_data/`](#ml01_data)
    - [`ML/02_notebook/`](#ml02_notebook)
    - [`ML/03_models/`](#ml03_models)
    - [`ML/04_reports/`](#ml04_reports)
    - [`ML/05_app_ready/`](#ml05_app_ready)
  - [Reproduce or retrain](#reproduce-or-retrain)
  - [Troubleshooting](#troubleshooting)

## Project at a glance

The application runs as two local processes:

1. A Flask service loads the trained models and exposes the input schema and prediction endpoints.
2. A React application renders the form from that schema, validates student profiles, and presents the API response.

The ML pipeline has two primary outputs:

- **Placement probability:** a calibrated classification estimate for whether a student is placed.
- **Salary if placed:** a regression estimate in LPA, accompanied by a lower-to-upper salary interval.

The predictor also calculates an expected LPA (`placement probability × salary if placed`) and a profile percentile. The UI presents placement chance and salary-if-placed separately; expected LPA is a derived combined estimate, not a promised salary.

## Repository structure

```text
placement_project/
├── README.md
├── HOW_TO_RUN.md
├── backend/
│   ├── app.py                     Flask API, validation, model loading
│   └── requirements.txt           Backend + pinned model dependencies
├── frontend/
│   ├── index.html                 Page metadata
│   ├── package.json               React and Vite scripts/dependencies
│   ├── vite.config.js             Dev server and /api proxy
│   └── src/
│       ├── App.jsx                Page sections, schema/model info, app state
│       ├── Form.jsx               Schema-driven form and client validation
│       ├── Result.jsx             Prediction results and reliability note
│       ├── api.js                 Backend requests and API errors
│       ├── format.js              LPA, rupee, and percentage formatting
│       ├── NotFound.jsx           Not-found view
│       └── styles.css             Responsive application styles
└── ML/
    ├── 01_data/                   Dataset CSV
    ├── 02_notebook/               End-to-end ML notebook
    ├── 03_models/                 Trained models, schema, model card
    ├── 04_reports/                Evaluation tables, metrics, figures
    └── 05_app_ready/              Predictor utility, dependencies, sample input
```

## Run locally

### Prerequisites

- Python **3.12** is recommended for the saved model artifacts. Install the exact versions listed in `ML/05_app_ready/requirements.txt`; changing scikit-learn/XGBoost versions can make serialized models incompatible.
- Node.js 18 or newer and npm.
- The checked-in model artifacts under `ML/03_models/` and predictor utility under `ML/05_app_ready/`.

Open two terminals from the repository root.

### 1. Start the backend (Windows PowerShell)

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

The API listens at `http://127.0.0.1:8000`. If you already have the project's compatible environment at `backend/.venv312`, activate that environment instead of creating `.venv`:

```powershell
cd backend
.\.venv312\Scripts\Activate.ps1
python app.py
```

The backend requirements include the pinned ML dependencies via `ML/05_app_ready/requirements.txt`, plus Flask.

### 2. Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite forwards `/api` requests to `http://127.0.0.1:8000`, so the browser uses the local Flask API without separate CORS configuration. Keep both processes running while using the app.

To create and preview a production build:

```powershell
cd frontend
npm run build
npm run preview
```

Vite writes the production files to `frontend/dist/`.

## How predictions work

The notebook builds and evaluates a two-stage prediction system:

1. **Placement classification** estimates the probability of placement from the student's profile. The selected model is Logistic Regression, with calibration applied in the current model card.
2. **Salary regression** estimates salary conditional on being placed. The selected model is XGBoost.
3. **Salary interval** uses separate lower- and upper-quantile models to produce a range around the salary estimate.
4. **Profile percentile** compares the predicted placement probability with stored training-set probability quantiles.

The model pipelines include a custom `FeatureEngineer` transformer. It creates `skill_index`, `experience_index`, and `has_backlog` from applicable input columns. The transformer definition lives in `ML/05_app_ready/ml_utils.py`; it must remain importable when loading the serialized pipelines. The backend adds that directory to Python's import path.

Training uses a train/test split and cross-validation, with preprocessing and feature engineering kept in pipelines. Age is excluded as a protected attribute. Exact model selection, calibration status, metrics, feature lists, and library versions are recorded in `ML/03_models/model_card.json` and `ML/03_models/feature_schema.json`.

## Input features

The form is generated from `ML/03_models/feature_schema.json`, served by `GET /api/schema`, and grouped in the frontend for readability. The schema defines each field's label, type, allowed range or categories, and default. The backend independently validates every required value before prediction.

The current input features are:

| Group | Fields |
|---|---|
| Academic | `cgpa`, `backlogs`, `attendance_percentage`, `academic_consistency`, `core_subject_score` |
| Technical | `technical_skills_count`, `coding_skill_score`, `dsa_score`, `github_projects`, `coding_contest_rating` |
| Experience and projects | `internship_count`, `internship_months`, `projects_count`, `work_experience_months`, `hackathon_count`, `certifications_count` |
| Communication and preparation | `communication_score`, `aptitude_score`, `resume_score`, `interview_score`, `placement_preparation_hours_per_week` |
| Activities and preferences | `extracurricular_score`, `college_tier`, `degree_type`, `specialization`, `preferred_job_role`, `location_preference` |

There are 27 model inputs. Numeric values must fall within the schema's ranges; integer fields must be whole numbers. Categorical values must match the choices supplied by the schema. The schema is the source of truth if this list and the model artifacts ever differ.

## Model performance and limitations

The following are held-out test metrics from the current `model_card.json` (100,000 rows in the recorded run). They are not guarantees of performance on new or real-world populations.

| Task | Current selected model | Test result |
|---|---|---|
| Placement classification | Logistic Regression (calibrated) | Accuracy 0.7356; precision 0.7825; recall 0.8120; F1 0.7970; ROC AUC 0.7981; PR AUC 0.8769; Brier score 0.1735 |
| Salary if placed | XGBoost | MAE 2.61 LPA; RMSE 3.83 LPA; R² 0.6558 |
| Salary interval | 10th/90th percentile models | 79.16% test coverage for the nominal 80% interval |
| Combined expected LPA | Probability × salary estimate | MAE 4.16 LPA; RMSE 5.16 LPA; R² 0.5887 |

The placement model card describes placement as only weakly predictable from the available features. Its held-out ROC AUC is 0.7981 (95% CI 0.7923–0.8045), and the estimated chances should be treated as a rough guide. Salary predictions are conditional on placement and also have substantial error; use the interval to communicate uncertainty. A high model score does not establish causation, fairness, or suitability for decisions about an individual.

For the latest authoritative numbers and library versions, inspect `ML/03_models/model_card.json` or `ML/04_reports/metrics_summary.json`. The report CSVs contain cross-validation comparisons and test-set results for candidate models.

## Backend API

The Flask app is implemented in `backend/app.py` and loads model artifacts once when it starts.

| Method and path | Purpose |
|---|---|
| `GET /api/schema` | Returns the model input fields, labels, types, ranges, and categorical options. |
| `GET /api/model-info` | Returns the training-row count and placement reliability note. |
| `POST /api/predict` | Validates a profile and returns a placement and salary prediction. |

Example request body: use `ML/05_app_ready/example_input.json`, which contains values valid for the current schema.

Example response shape (values are illustrative):

```json
{
  "placement_probability": 0.6304,
  "placement_percent": 63.0,
  "predicted_placed": true,
  "profile_percentile": 47.1,
  "salary_if_placed_lpa": 9.72,
  "salary_range_lpa": [7.37, 12.77],
  "salary_if_placed_inr": 972088,
  "salary_range_inr": [737092, 1277192],
  "expected_lpa": 6.13,
  "expected_inr": 612829
}
```

`1 LPA = INR 100,000 per year`. Invalid or incomplete profiles receive HTTP 400 with an `errors` object keyed by field name. The browser also validates inputs, but server-side validation is authoritative.

## Frontend

The existing React 18 + Vite application is kept in `frontend/`. It provides:

- A home/overview page and navigation for the predictor, methodology, model results, dataset, and source links.
- A form grouped by student-profile area and populated from the backend schema.
- Required-field, numeric range, whole-number, and categorical-option validation.
- Loading, success, and error states for API requests.
- Results for placement probability, predicted placement status, salary if placed, salary interval, profile percentile, and the model's reliability note.
- Responsive layouts and a not-found view.

The frontend does not contain a second copy of the model logic. It submits the schema-defined inputs to the backend and displays the returned prediction.

## ML artifacts and reports

### `ML/01_data/`

Contains the 100,000-row CSV used by the current project. The input columns, target definitions, preprocessing, and analysis are documented in the notebook. Treat the data as a demonstration dataset, not as a representative sample of all students or hiring outcomes.

### `ML/02_notebook/`

`ai-project04.ipynb` contains the machine-learning workflow, from data loading and exploration through feature engineering, model comparison, evaluation, explainability, and artifact generation.

### `ML/03_models/`

- `placement_classifier.joblib`: selected placement classification pipeline.
- `salary_regressor.joblib`: selected salary regression pipeline.
- `salary_q10.joblib` and `salary_q90.joblib`: lower- and upper-quantile salary models.
- `feature_schema.json`: model input definition used by the backend and frontend, including defaults and probability reference quantiles.
- `model_card.json`: selected models, test metrics, reliability summary, row count, and training library versions.

### `ML/04_reports/`

Contains cross-validation and test result tables for classification and regression, feature-selection and ablation tables, a metrics summary, and visualizations under `figures/`.

### `ML/05_app_ready/`

- `ml_utils.py`: `FeatureEngineer`, `PlacementPredictor`, prediction formatting, and prediction output construction.
- `requirements.txt`: exact core library versions recorded for the trained models.
- `example_input.json`: valid sample profile for API and predictor checks.
- `README.md`: short example of using `PlacementPredictor` directly.

## Reproduce or retrain

1. Review the setup and data-loading cells in `ML/02_notebook/ai-project04.ipynb` and ensure the expected CSV exists under `ML/01_data/` (or use the notebook's supported data source).
2. Install the notebook dependencies. The model-serving dependencies are pinned in `ML/05_app_ready/requirements.txt`; the notebook may require additional analysis packages.
3. Run the notebook cells in order. It documents the data processing and evaluation workflow and generates model artifacts and reports.
4. Check the generated schema and model card before serving the new artifacts.
5. Restart the backend so it loads the regenerated model files.

Do not mix a newly trained artifact set with an old schema or predictor utility. Keep `feature_schema.json`, model files, and `ml_utils.py` from the same notebook run together. For reproducibility details and notebook-specific controls, use the notebook itself as the authority.

## Troubleshooting

- **Frontend cannot reach the API:** start the backend first and confirm `http://127.0.0.1:8000/api/schema` responds. The Vite proxy forwards `/api` to this address.
- **Model loading error or corrupted XGBoost artifact:** check the Python version and install the exact pinned dependencies. Python 3.12 with the project's compatible environment has been used to serve the current artifacts; avoid installing a different XGBoost or scikit-learn version into that environment.
- **A prediction returns HTTP 400:** provide every field from `/api/schema`; use the supplied categorical options and numeric ranges. Integer fields cannot contain fractional values.
- **Port already in use:** stop the existing process using port 8000 (backend) or 5173 (Vite), or adjust the corresponding app configuration and proxy together.

---

For the short startup checklist, see [HOW_TO_RUN.md](HOW_TO_RUN.md).