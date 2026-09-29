# ML/05_app_ready - how to use the trained models

```python
import sys; sys.path.append("ML/05_app_ready")          # so joblib can find ml_utils.FeatureEngineer
from ml_utils import PlacementPredictor
predictor = PlacementPredictor("ML/03_models")            # loads classifier, regressor and quantile models
print(predictor.predict(json.load(open("ML/05_app_ready/example_input.json"))))
```

* `feature_schema.json` (in 03_models) lists every input field with type / min / max / default -> build the frontend form from it.
* Install the exact library versions first:  `pip install -r ML/05_app_ready/requirements.txt`
* Output keys: placement_percent, profile_percentile, salary_if_placed_lpa, salary_range_lpa, salary_if_placed_inr, expected_lpa ...
* 1 LPA = Rs 1,00,000 per year.
