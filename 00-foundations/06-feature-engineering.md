# Chapter 0.6 — Feature Engineering & Preprocessing Pipelines

[← Roadmap](../README.md) · Prev [0.5](05-evaluation.md) · Next [1.1 Linear Regression](../01-regression/01-linear-regression.md) · Priority 🔴

## 1. Why
Models see numbers, not meaning. Good features encode domain knowledge; for tabular data they often matter more than the algorithm.

## 2. Which models need what
| Step | Linear/SVM/k-NN/NN | Trees & boosting |
|---|---|---|
| Scaling | **Required** (distance/penalty/GD based) | Not needed |
| Missing values | Impute | LightGBM/XGBoost/HistGB handle natively |
| Categorical | One-hot / target / embedding | Ordinal OK; CatBoost native |
| Skew | log/Box-Cox helps | Invariant to monotonic transforms |
| Outliers | Clip/robust scale | Mostly robust |

## 3. Techniques
- **Scaling:** standardize `(x−μ)/σ`; min-max; robust (median/IQR). Fit on train only.
- **Missing data:** mean/median/mode, constant + *missing-indicator*, KNN/iterative imputation. Understand why it's missing (MCAR/MAR/MNAR).
- **Categorical:** one-hot (low cardinality), ordinal (true order), frequency/count, **target encoding** (must be out-of-fold with smoothing to avoid leakage), hashing, learned embeddings (high cardinality).
- **Numeric transforms:** log, sqrt, power, binning, splines, polynomial/interaction terms.
- **Dates:** hour, weekday, month, cyclic encodings `sin/cos(2πt/T)`, time since event.
- **Text:** bag-of-words/TF-IDF → embeddings. **Images/audio:** use pretrained encoders.
- **Aggregations:** per-user/group statistics (computed on past data only).
- **Imbalance:** class weights, threshold tuning, resampling (SMOTE inside CV folds only).
- **Dimensionality:** PCA, selection (Ch 4.7).

## 4. Worked example — standardization
Values `[10, 12, 14, 16, 48]`: mean 20, population std ≈ 14.1. The 48 becomes z ≈ +2.0 (and 10 → −0.71). With robust scaling (median 14, IQR 4): 48 → +8.5 — outliers stay visible instead of compressing everyone else; choose by whether outliers are signal or noise.

## 5. Pipeline pattern (leak-proof)
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

num, cat = ["age", "income"], ["city", "plan"]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
# cross_val_score(model, X, y, cv=5)  -> preprocessing refit inside each fold
```
Column names are illustrative; adapt to your data.

## 6. Pitfalls
Fitting scalers/encoders on all data · target leakage via aggregates that include the future · high-cardinality one-hot blow-up · train/serve skew (different code paths in training and production) — use one shared pipeline object · silently changing category sets.

## 7. Checklist
Does each feature exist at prediction time? Is it computed identically online and offline? Is it stable over time? Does removing it change the score (ablation)?

## Interview Q&A
- *Do trees need scaling?* No; splits depend only on ordering.
- *Target encoding risk?* Leakage/overfitting; use out-of-fold + smoothing.
- *How handle cyclic features?* Sine/cosine pair so 23:00 is near 00:00.

## Exercises
Add missing-indicators to a dataset and measure the change; implement out-of-fold target encoding and show leakage when done naively.

## Further reading
scikit-learn "Preprocessing data"; Kuhn & Johnson, *Feature Engineering and Selection* (free online); Zheng & Casari, *Feature Engineering for ML*.
