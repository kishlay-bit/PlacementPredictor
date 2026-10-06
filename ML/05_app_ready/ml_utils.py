import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

SKILL_COLS = ["coding_skill_score", "aptitude_score", "communication_score", "dsa_score", "core_subject_score"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    # Stateless feature engineering (nothing is learned from data -> cannot leak information).
    #   skill_index      = mean of the four skill scores
    #   experience_index = internships + projects
    #   has_backlog      = 1 if backlogs > 0 else 0
    def __init__(self, add_features=True):
        self.add_features = add_features

    def _new_cols(self, cols):
        cols = list(cols)
        new = []
        if self.add_features:
            if sum(c in cols for c in SKILL_COLS) >= 2:
                new.append("skill_index")
            if "internship_count" in cols and "projects_count" in cols:
                new.append("experience_index")
            if "backlogs" in cols:
                new.append("has_backlog")
        return new

    def fit(self, X, y=None):
        self.columns_in_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self._new_cols(X.columns):
            if c == "skill_index":
                X[c] = X[[s for s in SKILL_COLS if s in X.columns]].mean(axis=1)
            elif c == "experience_index":
                X[c] = X["internship_count"] + X["projects_count"]
            elif c == "has_backlog":
                X[c] = (X["backlogs"] > 0).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.columns_in_) + self._new_cols(self.columns_in_), dtype=object)


def format_inr(amount):
    # Indian digit grouping, e.g. 1350000 -> "Rs 13,50,000"
    s = str(int(round(amount)))
    if len(s) <= 3:
        return "Rs " + s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return "Rs " + ",".join(parts + [tail])


class PlacementPredictor:
    # Loads the saved models and returns placement chance + salary estimate for one student (dict) or many (DataFrame).
    def __init__(self, model_dir=None):
        d = Path(model_dir) if model_dir else Path(__file__).resolve().parent.parent / "03_models"
        self.clf = joblib.load(d / "placement_classifier.joblib")
        self.reg = joblib.load(d / "salary_regressor.joblib")
        self.q_lo = joblib.load(d / "salary_q10.joblib")
        self.q_hi = joblib.load(d / "salary_q90.joblib")
        with open(d / "feature_schema.json") as f:
            self.schema = json.load(f)
        self.pq = np.array(self.schema.get("probability_quantiles", []), dtype=float)

    def _frame(self, data):
        df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
        missing = [c for c in self.schema["input_features"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing input fields: {missing}")
        for col, meta in self.schema["features"].items():
            if meta["type"] in ("integer", "float"):
                df[col] = df[col].astype(float).clip(meta["min"], meta["max"])
        return df

    def predict_frame(self, data):
        df = self._frame(data)
        p = self.clf.predict_proba(df[self.schema["classifier_features"]])[:, 1]
        rf = self.schema["regressor_features"]
        sal = np.clip(self.reg.predict(df[rf]), 0, None)
        lo = np.clip(self.q_lo.predict(df[rf]), 0, None)
        hi = np.clip(self.q_hi.predict(df[rf]), 0, None)
        lo, hi = np.minimum(lo, sal), np.maximum(hi, sal)      # guarantee lo <= point <= hi
        pct = np.interp(p, self.pq, np.linspace(0, 100, len(self.pq))) if len(self.pq) > 1 else np.full(len(p), np.nan)
        return pd.DataFrame({"placement_probability": p, "profile_percentile": pct, "salary_if_placed_lpa": sal,
                             "salary_low_lpa": lo, "salary_high_lpa": hi,
                             "expected_lpa": p * sal})

    def predict(self, data):
        r = self.predict_frame(data).iloc[0]
        p, sal, lo, hi, exp = (float(r[k]) for k in ["placement_probability", "salary_if_placed_lpa",
                                                      "salary_low_lpa", "salary_high_lpa", "expected_lpa"])
        return {
            "profile_percentile": None if np.isnan(r["profile_percentile"]) else round(float(r["profile_percentile"]), 1),
            "placement_probability": round(p, 4),
            "placement_percent": round(100 * p, 1),
            "predicted_placed": bool(p >= 0.5),
            "salary_if_placed_lpa": round(sal, 2),
            "salary_range_lpa": [round(lo, 2), round(hi, 2)],
            "salary_if_placed_inr": int(round(sal * 1e5)),
            "salary_range_inr": [int(round(lo * 1e5)), int(round(hi * 1e5))],
            "salary_if_placed_text": format_inr(sal * 1e5) + " per year",
            "expected_lpa": round(exp, 2),
            "expected_inr": int(round(exp * 1e5)),
        }
