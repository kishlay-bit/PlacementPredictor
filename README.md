# Student Placement Prediction System

A machine-learning project that takes a student's academic and skill profile and estimates

1. the **chance of being placed** (in %), and
2. the **salary package if placed** (in LPA and in rupees per year, with a likely range).

The project has three parts: a **machine-learning pipeline** (`ML/`, the core of the work), a small **Python backend** (`backend/`) that serves the trained models, and a **React frontend** (`frontend/`) with a single form and a result panel. Everything runs on localhost.

Most of this document is about the ML part, because that is where the decisions are.

---

## Contents

1. [Project structure](#1-project-structure)
2. [Quick start](#2-quick-start)
3. [The dataset](#3-the-dataset)
4. [ML design: two stages](#4-ml-design-two-stages)
5. [Avoiding data leakage](#5-avoiding-data-leakage)
6. [Exploratory analysis](#6-exploratory-analysis)
7. [Feature selection: which columns we kept and why](#7-feature-selection-which-columns-we-kept-and-why)
8. [Preprocessing: transformation and encoding](#8-preprocessing-transformation-and-encoding)
9. [The models](#9-the-models)
10. [Why we compare models, and how we choose](#10-why-we-compare-models-and-how-we-choose)
11. [Results](#11-results)
12. [Extra techniques used](#12-extra-techniques-used)
13. [Explainability](#13-explainability)
14. [Limitations and honest interpretation](#14-limitations-and-honest-interpretation)
15. [Saved artifacts](#15-saved-artifacts)
16. [Backend](#16-backend)
17. [Frontend](#17-frontend)
18. [Retraining and reproducibility](#18-retraining-and-reproducibility)
19. [FAQ](#19-faq)

---

## 1. Project structure

```
placement_project/
├── README.md                this file
├── HOW_TO_RUN.md            short run instructions
├── backend/
│   ├── app.py               Flask API that loads the trained models
│   └── requirements.txt     Python dependencies for the backend
├── frontend/
│   ├── index.html
│   ├── package.json         React dependencies (npm install)
│   ├── vite.config.js       dev server + proxy to the backend
│   └── src/                 App, Form, Result, API helper, styles
└── ML/
    ├── 01_data/             raw CSV
    ├── 02_notebook/         the full pipeline as one notebook
    ├── 03_models/           trained models + feature schema + model card
    ├── 04_reports/          metric tables and figures
    └── 05_app_ready/        ml_utils.py (predictor class), requirements, example input
```

The `ML` folders are numbered in the order the work happens: data, then notebook, then models, then reports, then the files the app needs.

## 2. Quick start

Prerequisite: the notebook has been run, so `ML/03_models` and `ML/05_app_ready` exist.

**Backend** (Python 3.10 or newer)

```
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
python app.py
```

The API runs on http://127.0.0.1:8000. `requirements.txt` reuses the exact library versions the models were trained with (`ML/05_app_ready/requirements.txt`) and adds Flask. Matching versions matter because the models are saved as pickled scikit-learn pipelines.

**Frontend** (Node 18 or newer)

```
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Start the backend first, otherwise the page shows a "could not reach the prediction server" message.

---

## 3. The dataset

| Item | Value |
|---|---|
| File | `ML/01_data/student_placement_prediction_dataset_2026.csv` (Kaggle: `kishlaytejeswi/student-placement-prediction-dataset`) |
| Size | 100,000 students, 26 columns |
| Missing values | none |
| Duplicate rows | none (`student_id` is unique) |
| Placement rate | 54.46 % placed |
| Salary of placed students | mean 13.32 LPA, standard deviation 1.59, range 7.11 to 20.44 |

The columns fall into these groups:

- **Identifier:** `student_id`
- **Demographics:** `age`, `gender`
- **Academic background:** `cgpa`, `branch`, `college_tier`, `backlogs`, `attendance_percentage`
- **Experience:** `internships_count`, `projects_count`, `certifications_count`, `hackathons_participated`, `github_repos`, `linkedin_connections`
- **Skill scores:** `coding_skill_score`, `aptitude_score`, `communication_skill_score`, `logical_reasoning_score`, `mock_interview_score`
- **Activities and lifestyle:** `extracurricular_score`, `leadership_score`, `volunteer_experience`, `sleep_hours`, `study_hours_per_day`
- **Outcomes (targets):** `placement_status`, `salary_package_lpa`

One structural fact drives the design: `salary_package_lpa > 0` if and only if the student is placed. The notebook checks this explicitly. Unplaced students have a salary of 0, which is not a real salary, so it must not be used to train a salary model (see the next section).

Our reading of the data is that it is very likely **synthetic** (uniform-looking distributions, and most columns having almost no relationship with the outcomes). The dataset page does not confirm this; it is an inference from the analysis below. It matters for how the results should be read (see [section 14](#14-limitations-and-honest-interpretation)).

---

## 4. ML design: two stages

We do not train one model to predict "LPA" directly. Two questions are being asked, and they are different kinds of problem:

| Stage | Question | Type | Trained on | Output |
|---|---|---|---|---|
| 1 | Will this student be placed? | Binary classification | all students | P(placed) |
| 2 | If placed, what will the package be? | Regression | **placed students only** | salary in LPA, plus a 10th to 90th percentile range |

**Why placed students only for stage 2.** If unplaced students (salary 0) were included, the regression would try to average zeros with real salaries of 7 to 20 LPA. It would produce a number that describes nobody and it would blur the real relationship between skills and salary. Splitting the problem keeps each model honest about what it predicts.

**Why the app shows two numbers.** The student sees the chance of placement and the salary if placed. We deliberately do not headline "expected LPA" (chance multiplied by salary). It is a mathematically valid average, but it lands between 0 and about 13 and matches no real outcome. On the test set it barely beats guessing the average for everyone (see [section 11](#114-end-to-end-check)). The number is still computed and returned by the predictor, but the frontend does not show it.

---

## 5. Avoiding data leakage

Data leakage means information from the test data (or from the answer itself) sneaks into training, which makes results look better than they are. The pipeline guards against it in these ways:

1. **Split first.** The very first modelling step is a stratified 80/20 train/test split (80,000 and 20,000 rows; both keep the 54.46 % placement rate). The test set is then locked away. Nothing "learned" from data touches it until the final evaluation.
2. **Outcomes are never inputs.** `placement_status` and `salary_package_lpa` are the answers. They are excluded from the feature list in code.
3. **Identifier removed.** `student_id` carries no information and could let a flexible model memorise rows.
4. **Feature selection uses the training set only.** All statistical tests (section 7) are computed on training rows.
5. **Preprocessing lives inside a scikit-learn `Pipeline`.** Imputation, scaling and one-hot encoding are refitted on the training part of every cross-validation fold, never on the validation part.
6. **Hyper-parameter tuning uses the training set only**, with inner cross-validation, on a 30,000-row training subsample for speed.
7. **Cross-validation is inside the training set.** Five folds (stratified for classification) produce the mean and standard deviation used to compare models.
8. **The test set is used once**, after all choices are made.
9. **The calibration decision is made on training data** (out-of-fold predictions), not by peeking at the test set.
10. **Protected attributes** (`gender`, `age`) are excluded on ethical grounds, whatever the data says.

The model that ends up deployed is fitted on the training split only. This is a deliberate choice so the reported test numbers describe exactly the model the app serves.

---

## 6. Exploratory analysis

All exploratory work uses **training data only**.

### Targets

![Target distributions](ML/04_reports/figures/01_target_distributions.png)

The classes are nearly balanced (54 % placed), so no oversampling such as SMOTE is needed. Salaries of placed students cluster around 13 LPA.

### Correlations

![Correlation matrix](ML/04_reports/figures/02_correlation_matrix.png)

Correlation with the two targets (training data):

| Column | Correlation with placed | Correlation with salary (placed only) |
|---|---|---|
| backlogs | -0.085 | 0.008 |
| internships_count | 0.062 | 0.390 |
| projects_count | 0.060 | -0.009 |
| coding_skill_score | 0.051 | 0.455 |
| mock_interview_score | 0.046 | -0.005 |
| logical_reasoning_score | 0.036 | -0.001 |
| aptitude_score | 0.032 | 0.000 |
| communication_skill_score | 0.030 | 0.000 |
| leadership_score | 0.023 | 0.004 |
| extracurricular_score | 0.020 | 0.009 |
| cgpa | 0.014 | **0.503** |
| attendance_percentage | -0.010 | 0.002 |
| sleep_hours | -0.007 | 0.003 |
| certifications_count | -0.006 | 0.001 |
| github_repos | 0.005 | 0.001 |
| hackathons_participated | 0.004 | -0.003 |
| linkedin_connections | 0.004 | 0.004 |
| age | 0.003 | -0.003 |
| study_hours_per_day | -0.001 | 0.000 |

Two things stand out:

- **Salary is strongly structured.** Three columns (CGPA 0.50, coding skill 0.46, internships 0.39) carry all of the relationship. Everything else is essentially zero.
- **Placement is weakly structured.** The strongest single correlation is only -0.085 (backlogs). Every relationship is small.

The next figures show the strongest drivers visually.

![Top features by placement status](ML/04_reports/figures/03_top_features_by_status.png)

![Salary drivers](ML/04_reports/figures/04_salary_drivers.png)

---

## 7. Feature selection: which columns we kept and why

The goal was to keep only columns that have a real, measurable relationship with the target, and to drop the rest. This is **data reduction**, and it is done separately for each stage because the two targets depend on different columns.

### 7.1 The rules, in order

1. **Drop the identifier** (`student_id`).
2. **Drop protected attributes** (`gender`, `age`) on ethical grounds. A placement tool should not score people using them.
3. **Test every remaining column against each target**, on training data only:
   - numeric column vs placed: ANOVA F-test
   - categorical column vs placed: chi-squared test
   - numeric column vs salary: Pearson F-test
   - categorical column vs salary: one-way ANOVA
   A column is kept only if its p-value is below **0.001**. (There are about 20 tests, so this is stricter than the usual 0.05.)
4. **Mutual information** is computed as a second opinion, because it also catches non-linear relationships.
5. **Effect size** is reported next to the p-value (correlation, Cramer's V, or eta), with a label from negligible to strong.
6. **Ablation check** (section 7.4): confirm that dropping columns did not hurt performance.

### 7.2 What was kept

| Stage | Columns kept | Count |
|---|---|---|
| Placement classifier | cgpa, internships_count, projects_count, coding_skill_score, aptitude_score, communication_skill_score, logical_reasoning_score, mock_interview_score, backlogs, extracurricular_score, leadership_score | 11 of 21 |
| Salary regressor | cgpa, internships_count, coding_skill_score | 3 of 21 |

The form in the app shows the 11 columns the classifier uses (the salary columns are a subset of them), and all of them are mandatory.

### 7.3 What was dropped, and why

**Dropped from both stages:** `branch`, `college_tier`, `certifications_count`, `hackathons_participated`, `github_repos`, `linkedin_connections`, `attendance_percentage`, `volunteer_experience`, `sleep_hours`, `study_hours_per_day` (plus `gender`, `age` on ethical grounds).

**Dropped from salary only:** `projects_count`, `aptitude_score`, `communication_skill_score`, `logical_reasoning_score`, `mock_interview_score`, `backlogs`, `extracurricular_score`, `leadership_score`.

The reasoning matches your intuition that sleep hours should not decide placement. Sleep, study hours, attendance, GitHub repositories, LinkedIn connections and hackathons all showed essentially no relationship with either target (correlations around 0.00 to 0.01 and failing the significance test).

**"But surely college tier, branch and certifications matter?"** In real life they often do. In this dataset they do not, and a model can only learn what the data contains. We therefore follow the data and do not assume. Two pieces of evidence support dropping them:

- The statistical tests found no reliable relationship.
- The ablation below shows that adding them back changes nothing.

If real placement data showed an effect of college tier, the same pipeline would keep the column automatically. The rule is not "drop these columns"; the rule is "keep a column only if the data supports it".

A note on the classifier's kept list: with 80,000 training rows, even a tiny effect passes a significance test. Columns such as aptitude, communication and leadership were kept because they passed the test, but their effects are very small (correlations of 0.02 to 0.03). The ablation shows that keeping or dropping them does not change accuracy. This is why we report effect size next to the p-value: significance says an effect exists, effect size says how big it is.

### 7.4 Ablation: did dropping columns hurt?

A plain Logistic Regression (classification) and Ridge (regression) were cross-validated on the training set with three feature sets:

| Task | All 21 candidates | Selected only | Selected + engineered |
|---|---|---|---|
| Placement (ROC-AUC) | 0.5852 | 0.5856 | 0.5856 |
| Salary (R²) | 0.6090 | 0.6092 | 0.6092 |

The selected set is as good as the full set (marginally better), so the dropped columns were noise. The engineered features (`skill_index`, `experience_index`, `has_backlog`, see next section) gave no improvement, so the pipeline does **not** use them. They were tested rather than assumed.

Full tables: `ML/04_reports/feature_selection_table.csv` and `ML/04_reports/ablation_feature_sets.csv`.

---

## 8. Preprocessing: transformation and encoding

Everything below is inside one pipeline so it is refitted per fold (no leakage):

```
FeatureEngineer  ->  ColumnTransformer  ->  model
```

| Step | What it does | Applied to |
|---|---|---|
| Median imputation | fills any missing value with the median (the data has none now, but this keeps the app safe for future input) | numeric columns |
| Standard scaling | rescales to zero mean and unit variance, so linear models treat columns fairly | numeric columns |
| Most-frequent imputation | fills any missing category with the most common one | categorical columns |
| One-hot encoding | turns each category into 0/1 columns; unknown categories are ignored instead of crashing | categorical columns |
| Feature engineering (optional) | `skill_index` (mean of four skill scores), `experience_index` (internships plus projects), `has_backlog` (backlogs above 0). Stateless, so it cannot leak. | tested by ablation, switched off because it did not help |

In the final selected feature sets all columns are numeric, so scaling is the step that matters; the categorical branch stays in the pipeline for safety and for retraining with a different selection.

---

## 9. The models

Four models are compared for each stage, ordered from simplest to most advanced.

### 9.1 Classification (stage 1)

| # | Model | What it is | Why it is included |
|---|---|---|---|
| 1 | **Logistic Regression** | A linear model that turns a weighted sum of the inputs into a probability | The baseline. Simple, fast, hard to overfit, and its probabilities are usually well calibrated |
| 2 | **Random Forest** | Many decision trees trained on random samples and averaged (bagging) | Captures non-linear effects and interactions between columns |
| 3 | **XGBoost** | Trees added one after another, each correcting the errors of the previous ones (gradient boosting) | Usually the strongest single model on tabular data |
| 4 | **Stacking Ensemble** | Logistic Regression, Random Forest, XGBoost and LightGBM make predictions; a Logistic Regression "meta-model" learns how to combine them | The most advanced option. Uses each model's strengths |

### 9.2 Regression (stage 2)

Same ladder: **Ridge Regression** (linear with a penalty against over-large weights), **Random Forest**, **XGBoost**, **Stacking** (Ridge, Random Forest, XGBoost, LightGBM, combined by a Ridge meta-model).

### 9.3 Tuning

- Logistic Regression and Ridge: grid search over the regularisation strength.
- Random Forest and XGBoost: randomized search (10 iterations) over tree count, depth, leaf size, learning rate, subsampling and regularisation.
- Search is done with 3-fold inner cross-validation on a 30,000-row training subsample, for speed. The tuned models are then evaluated on the full training set.
- Settings the search chose: Logistic Regression `C = 0.01`; Ridge `alpha = 1`; Random Forest classifier 250 trees, depth 8, leaf size 100; XGBoost depth 2, learning rate 0.03, 400 trees. The search preferring shallow trees, large leaves and strong regularisation is itself a sign that the data are noisy: flexible models are punished for chasing noise.
- LightGBM is used only as an extra, differently-built base learner inside the stack, with fixed regularised settings.

---

## 10. Why we compare models, and how we choose

There is no way to know in advance which model suits a dataset. Comparing them answers a question with evidence instead of guessing. The comparison is designed to be fair:

- **Same data, same folds.** Every model sees the same cross-validation splits.
- **A "no skill" reference.** A dummy model (always predict the average) sets the floor. A model is only interesting if it beats it.
- **Mean and spread.** Each model is scored on five folds, and we look at the mean and the standard deviation. A difference smaller than the spread is noise.
- **Train score vs validation score.** A large gap means overfitting. For example, the Random Forest classifier scores 0.626 AUC on the data it trained on but 0.581 on held-out folds; Logistic Regression scores 0.586 on both.
- **Several metrics.** Accuracy, precision, recall, F1, ROC-AUC and Brier score for classification; MAE, RMSE and R² for regression.
- **One-standard-error rule for selection.** Among all models whose score is within one standard deviation of the best, choose the **simplest**. If a complex model is only statistically tied with a simple one, its complexity buys nothing and adds risk, cost and slower predictions.
- **Test set last.** After the choice, all four models are refitted and scored once on the untouched test set, so the final comparison is reported honestly.

---

## 11. Results

The numbers below come from the run on the full 100,000-row dataset with seed 42. The exact figures for your latest run are always in `ML/04_reports/` and `ML/03_models/model_card.json`.

### 11.1 Stage 1: placement classification (5-fold CV on training data)

| Model | ROC-AUC (mean ± std) | Accuracy | Precision | Recall | F1 | Brier (lower is better) | Train AUC |
|---|---|---|---|---|---|---|---|
| Always predict "placed" (reference) | 0.500 | 0.5446 | 0.5446 | 1.000 | 0.705 | 0.2480 | 0.500 |
| 1. Logistic Regression | 0.5856 ± 0.0045 | 0.5700 | 0.5785 | 0.7754 | 0.6626 | 0.2422 | 0.5862 |
| 2. Random Forest | 0.5814 ± 0.0037 | 0.5679 | 0.5699 | 0.8422 | 0.6798 | 0.2432 | 0.6261 |
| 3. XGBoost | 0.5829 ± 0.0041 | 0.5689 | 0.5735 | 0.8133 | 0.6727 | 0.2427 | 0.5951 |
| 4. Stacking Ensemble | 0.5855 ± 0.0044 | 0.5697 | 0.5780 | 0.7777 | 0.6631 | 0.2422 | 0.5861 |

![Cross-validated ROC-AUC](ML/04_reports/figures/05_cls_cv_auc.png)

**Selected: Logistic Regression** (one-standard-error rule). All four models are within noise of each other, so the simplest one wins.

On the untouched test set the selected model reaches ROC-AUC of about **0.586** and accuracy of about **0.57**, against 0.5446 accuracy for always answering "placed".

![ROC curves and confusion matrix](ML/04_reports/figures/06_cls_roc_confusion.png)

**How to read this honestly.** An AUC of 0.586 means the model ranks a randomly chosen placed student above a randomly chosen unplaced student about 59 % of the time, where 50 % is a coin flip. That is a real but **weak** signal. Two facts support that this is a limit of the data, not of the modelling:

- Four very different models, from linear to boosted ensembles, land at the same score.
- Adding every dropped column back changed nothing (section 7.4).

If a non-linear pattern existed, the tree models would have found it and beaten Logistic Regression. They did not.

### 11.2 Probability quality (calibration)

A model's raw score is not automatically a probability. Calibration checks whether "60 %" really means that about 60 % of such students were placed.

![Calibration curve](ML/04_reports/figures/07_calibration.png)

The notebook tests whether an isotonic calibration wrapper improves the Brier score on training-set out-of-fold predictions, and applies it **only if it does**. In the original run it made no practical difference (test Brier 0.2422 before, 0.2424 after), which is expected because Logistic Regression is already well calibrated. Whether it was applied in your latest run is recorded as `calibration_applied` in `model_card.json`.

### 11.3 Stage 2: salary regression (placed students, 5-fold CV)

| Model | MAE (LPA) | RMSE (LPA) | R² | Train R² | Fit time |
|---|---|---|---|---|---|
| Always predict the mean (reference) | 1.277 | 1.594 | -0.000 | 0.000 | 0 s |
| 1. Ridge Regression | 0.797 | 0.9967 ± 0.0057 | 0.6092 | 0.6093 | 0.01 s |
| 2. Random Forest | 0.806 | 1.0084 ± 0.0066 | 0.5999 | 0.6241 | 3.9 s |
| 3. XGBoost | 0.799 | 0.9998 ± 0.0057 | 0.6068 | 0.6100 | 0.2 s |
| 4. Stacking Ensemble | 0.797 | 0.9967 ± 0.0057 | 0.6092 | 0.6085 | 14.1 s |

**Selected: Ridge Regression.** Stacking ties it exactly but takes about a thousand times longer to fit, so the simplest model wins. On the test set: **MAE 0.80 LPA, RMSE 1.00 LPA, R² 0.60**. In practice the salary estimate is typically within about 0.8 LPA of the true value, against 1.28 LPA for guessing the average.

![Predicted vs actual salary](ML/04_reports/figures/08_reg_pred_vs_actual.png)

That a plain linear model matches boosted ensembles says the salary relationship is essentially linear in CGPA, coding skill and internships, plus noise that no model can remove.

### 11.4 End-to-end check

Combining both stages as `expected LPA = P(placed) × salary if placed` and comparing with what students actually earned (0 if unplaced):

| System | MAE (LPA) | RMSE (LPA) | R² |
|---|---|---|---|
| Expected-LPA system | 6.40 | 6.58 | 0.045 |
| Naive: predict the training mean for everyone | 6.60 | 6.73 | 0.000 |

The combined system is only slightly better than the naive guess. This is the direct consequence of the weak placement stage: whether a student is placed at all cannot be predicted well from these columns. That is why the app leads with the salary-if-placed figure, which is the reliable output.

![Probability separation](ML/04_reports/figures/09_probability_separation.png)

---

## 12. Extra techniques used

These go beyond a plain "train four models" project.

| Technique | What it does | Why it is there |
|---|---|---|
| Leakage-safe split-first workflow | Split, then select, tune and fit only on training data | Honest test results |
| Statistical feature selection with effect sizes | Keeps a column only when the data supports it and reports how strong the relationship is | Answers "is this attribute actually related?" with evidence |
| Fairness exclusion | `gender` and `age` never used | Ethical use of a scoring tool |
| Ablation testing | Engineered features and dropped columns are tested, not assumed | Avoids complexity that does not help |
| One-standard-error selection | Simplest model within noise of the best | Prevents choosing complexity for no gain |
| Conditional calibration | Isotonic calibration applied only if it improves out-of-fold Brier score | Honest, well-behaved probabilities |
| **Quantile regression for a salary range** | Two LightGBM models estimate the 10th and 90th percentile of salary | A single number hides uncertainty. The result is an 80 % range; on the test set it covered 79.3 % of students (target 80 %), with an average width of 2.54 LPA |
| Placement honesty check | Bootstrap confidence interval for AUC, placement rate of the top and bottom 10 % of predictions, and how wide the predicted chances spread | Puts a number on how weak the placement signal is |
| **Profile percentile** | The app also reports how a profile ranks among the training students | The raw placement chance only spans a narrow band, so the percentile is a more informative way to compare students |
| What-if analysis | Varies one input at a time for an average student | Shows how the prediction moves and sanity-checks the model |
| Reusable predictor class | `PlacementPredictor` loads all models and returns one clean JSON result; inputs are clipped to the training range | One tested code path used by the backend |

![What-if analysis](ML/04_reports/figures/12_what_if.png)

![Decile lift on the test set](ML/04_reports/figures/13_decile_lift.png)

The decile chart splits test students into ten groups by predicted probability and shows the real placement rate in each. A useful model would slope clearly upward; the size of the slope is the honest measure of how much the placement model helps.

---

## 13. Explainability

Two methods show why the models produce their outputs:

- **Permutation importance:** shuffle one column and measure how much the score drops. Works for any model.
- **SHAP:** shows the direction and size of each column's push on individual predictions (computed on the XGBoost models).

![Permutation importance](ML/04_reports/figures/10_permutation_importance.png)

![SHAP for placement](ML/04_reports/figures/11_shap_placement.png)

![SHAP for salary](ML/04_reports/figures/11_shap_salary.png)

Consistent with the correlation analysis, salary is driven by CGPA, coding skill and internships, while placement importance is spread thinly over many columns.

---

## 14. Limitations and honest interpretation

- **Placement is only weakly predictable here.** ROC-AUC is about 0.59. Predicted chances fall in a narrow band, so a "60 %" and a "55 %" are hard to tell apart. Use the percentile and the reliability note in the app to interpret them.
- **Salary is the reliable output** (R² about 0.60, typical error about 0.8 LPA), and even that leaves 40 % of the variation unexplained.
- **The dataset appears to be synthetic.** Real campus placement depends on factors this data does not capture (the hiring market, company, city, interview performance on the day). The tool describes patterns in this dataset, not the real job market.
- **Relationships are correlations, not causes.** Doing more internships is associated with a higher predicted salary here; the model does not prove it would cause one.
- **Dropped columns are dropped for this data.** Their removal does not mean they are irrelevant in reality.
- **Protected attributes are excluded, but proxies can exist.** Removing `gender` and `age` avoids direct use; it does not prove the model is free of indirect bias.
- **This is a class project.** It must not be used for real decisions about real students.

---

## 15. Saved artifacts

**`ML/03_models/`**

| File | Content |
|---|---|
| `placement_classifier.joblib` | The selected classifier pipeline (preprocessing plus model) |
| `salary_regressor.joblib` | The selected salary regressor pipeline |
| `salary_q10.joblib`, `salary_q90.joblib` | Quantile models for the low and high end of the salary range |
| `feature_schema.json` | Every input field with its type, minimum, maximum and label, plus the reference table used for the profile percentile. The frontend form is built from this |
| `model_card.json` | Selected models, test metrics, calibration decision, range coverage, the honesty-check summary and library versions |

**`ML/04_reports/`**

| File | Content |
|---|---|
| `feature_selection_table.csv` | p-values, mutual information, effect sizes and keep/drop decision per column |
| `ablation_feature_sets.csv` | The three-way feature-set comparison |
| `classification_cv_results.csv`, `regression_cv_results.csv` | Cross-validation comparison of the four models per stage |
| `classification_test_results.csv`, `regression_test_results.csv` | Test-set results for all four models |
| `metrics_summary.json` | Copy of the model card |
| `figures/` | All charts referenced in this document |

**`ML/05_app_ready/`**

| File | Content |
|---|---|
| `ml_utils.py` | `FeatureEngineer` (required to load the saved pipelines) and `PlacementPredictor` |
| `requirements.txt` | Exact library versions used for training |
| `example_input.json` | A sample input |
| `README.md` | Short usage note for the predictor |

---

## 16. Backend

`backend/app.py` is a small Flask application. It loads the predictor once at start-up and exposes three endpoints on `http://127.0.0.1:8000`.

| Method and path | Purpose |
|---|---|
| `GET /api/schema` | The list of input fields (name, label, type, min, max, options) used to build the form |
| `GET /api/model-info` | Number of training rows and the reliability note shown under the result |
| `POST /api/predict` | Validates the input and returns the prediction |

**Validation.** Every field the models need must be present, numeric where required, inside the training range, and a whole number where the column is an integer. On any problem the server returns HTTP 400 with an `errors` object naming each field. The browser validates too, but the server never trusts the browser.

**Prediction response** (shape; values illustrative):

```json
{
  "placement_percent": 67.4,
  "placement_probability": 0.6736,
  "predicted_placed": true,
  "profile_percentile": 84.0,
  "salary_if_placed_lpa": 14.7,
  "salary_range_lpa": [13.51, 16.01],
  "salary_if_placed_inr": 1470395,
  "salary_range_inr": [1351237, 1600606],
  "salary_if_placed_text": "Rs 14,70,395 per year",
  "expected_lpa": 9.91,
  "expected_inr": 990514
}
```

`1 LPA = Rs 1,00,000 per year`.

**How the models load.** The saved pipelines contain the custom `FeatureEngineer` class, so Python can only unpickle them if `ml_utils.py` is importable. The backend adds `ML/05_app_ready` to the import path for this reason. This is also why the library versions in `backend/requirements.txt` must match training.

---

## 17. Frontend

A React 18 application built with Vite, using plain CSS. There are no UI libraries, so the page stays simple.

**What the page does**

- One column with the input fields, grouped into Academics, Experience, Skill scores and Activities.
- **Every field is mandatory.** Each shows its allowed range. Empty, out-of-range or non-whole values are flagged next to the field, and the first bad field receives focus.
- After **Get estimate**, the result appears below the form: the chance of placement with a bar, where the profile ranks among the training students, the estimated package if placed in LPA and rupees, the likely range, and the reliability note.
- **Clear** empties the form and hides the result.
- If the backend is not running, the page says so instead of failing silently.

**How it is built**

| File | Role |
|---|---|
| `src/App.jsx` | Loads the schema and model info, holds the result, scrolls to it |
| `src/Form.jsx` | Builds the form from the schema, validates, submits |
| `src/Result.jsx` | Displays the prediction |
| `src/api.js` | The three API calls and error handling |
| `src/format.js` | Rupee (Indian digit grouping) and LPA formatting |
| `src/styles.css` | Neutral styling: system font, one muted colour, no gradients or icons |
| `vite.config.js` | Dev server on port 5173 that forwards `/api` requests to the backend, so no cross-origin setup is needed |

Because the form is generated from `feature_schema.json`, retraining with a different feature selection changes the form automatically; no frontend code has to be edited.

---

## 18. Retraining and reproducibility

1. Put the CSV in `ML/01_data/` (on Kaggle the notebook finds it automatically).
2. Open `ML/02_notebook/placement_predication.ipynb` and run all cells. It rebuilds `03_models`, `04_reports` and `05_app_ready` and creates a zip of the whole project.
3. Restart the backend so it loads the new models.

Notes:

- Every random step uses seed **42**, so results reproduce.
- Setting the environment variable `QUICK_MODE=1` runs a small, fast test version on a 20,000-row sample. Its numbers are not meaningful; use it only to check that the code runs.
- A full run takes several minutes, mostly for the Random Forest search and the stacking models.
- Every notebook cell starts with a tag comment that says what it does, for example `[DATA REDUCTION]`, `[TRANSFORMATION + ENCODING]`, `[EVALUATION - TEST SET, used once]`.

---

## 19. FAQ

**Why is the placement chance always somewhere around 45 to 65 %?**
Because the columns carry little information about placement. The model cannot separate students strongly, so its probabilities stay close to the overall placement rate. The profile percentile shows how a student compares with others.

**Why not use a neural network?**
For tabular data of this size, gradient-boosted trees and linear models are the standard strong choices, and the results show the data has no complex pattern for a neural network to find. A bigger model would only fit the noise.

**Why did the simplest model win?**
Because all models performed the same within noise. The one-standard-error rule prefers the simpler model in that case: it is faster, easier to explain, and less likely to overfit.

**Why not include sleep hours, college tier and so on?**
The data shows no relationship between them and the targets, and adding them back did not improve any score. See section 7.

**Can the app be deployed?**
It is set up for localhost. Deploying would need the backend hosted, a production server instead of Flask's built-in one, and a build of the frontend (`npm run build`).

---

## Disclaimer

This project is for learning and demonstration. The predictions describe patterns in one dataset and are not a guarantee or a recommendation about any real student.
