"""Linear regression with scikit-learn: pipeline, CV, baseline, residual checks."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv = KFold(5, shuffle=True, random_state=0)
models = {
    "baseline (mean)": DummyRegressor(strategy="mean"),
    "linear": make_pipeline(StandardScaler(), LinearRegression()),
    "linear + log(y)": TransformedTargetRegressor(
        make_pipeline(StandardScaler(), LinearRegression()), func=np.log, inverse_func=np.exp),
}
for name, m in models.items():
    s = cross_validate(m, X_train, y_train, cv=cv,
                       scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"])
    print(f"{name:16s} R2={s['test_r2'].mean():.3f}  RMSE={-s['test_neg_root_mean_squared_error'].mean():.3f}"
          f"  MAE={-s['test_neg_mean_absolute_error'].mean():.3f}")

# Final, one-time test-set evaluation of the chosen model
final = models["linear"].fit(X_train, y_train)
pred = final.predict(X_test)
print(f"\nTEST  R2={r2_score(y_test, pred):.3f}  RMSE={mean_squared_error(y_test, pred) ** 0.5:.3f}"
      f"  MAE={mean_absolute_error(y_test, pred):.3f}")
coefs = final[-1].coef_
for n, c in sorted(zip(X.columns, coefs), key=lambda t: -abs(t[1])):
    print(f"  {n:11s} {c:+.3f}  (per 1 std-dev, in $100k)")
