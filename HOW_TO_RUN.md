# Running the app on localhost

Two processes: a Python backend (loads the trained models) and the React frontend. Run the ML notebook first so that `ML/03_models` and `ML/05_app_ready` exist.

## 1. Backend (Python 3.10+)

```
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python app.py
```

Runs on http://127.0.0.1:8000. `requirements.txt` reuses the exact library versions the models were trained with (`ML/05_app_ready/requirements.txt`) and adds Flask.

## 2. Frontend (Node 18+)

```
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. `npm install` reads the dependencies from `frontend/package.json`.

## Notes
- The form fields come from `ML/03_models/feature_schema.json`, so if you retrain and the selected features change, the form updates by itself.
- Every field is mandatory and range-checked in the browser and again on the server.
