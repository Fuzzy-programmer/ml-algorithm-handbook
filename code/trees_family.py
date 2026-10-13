"""Trees -> forests -> boosting -> XGBoost/LightGBM/CatBoost/HistGB on California housing + adult-like classification."""
import time, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
                              BaggingRegressor, AdaBoostRegressor, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error
import xgboost as xgb, lightgbm as lgb, catboost as cb

X, y = fetch_california_housing(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
rm = lambda m: float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))
models = {
 "Linear (Ch1.1)": LinearRegression(),
 "Tree depth=None": DecisionTreeRegressor(random_state=0),
 "Tree depth=6": DecisionTreeRegressor(max_depth=6, random_state=0),
 "Bagging(50 deep trees)": BaggingRegressor(DecisionTreeRegressor(), n_estimators=50, n_jobs=1, random_state=0),
 "RandomForest(200)": RandomForestRegressor(200, n_jobs=2, random_state=0),
 "ExtraTrees(200)": ExtraTreesRegressor(200, n_jobs=2, random_state=0),
 "AdaBoost(100)": AdaBoostRegressor(n_estimators=100, random_state=0),
 "sklearn GBM(300,d3)": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.1, random_state=0),
 "HistGB": HistGradientBoostingRegressor(random_state=0),
 "XGBoost": xgb.XGBRegressor(n_estimators=500, learning_rate=.05, max_depth=6, subsample=.8, colsample_bytree=.8, n_jobs=-1, random_state=0),
 "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=.05, num_leaves=31, subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, random_state=0),
 "CatBoost": cb.CatBoostRegressor(iterations=500, learning_rate=.08, depth=6, verbose=0, random_seed=0),
}
print("== California housing, test RMSE (units $100k) and fit time ==")
for n, m in models.items():
    t = time.time(); m.fit(Xtr, ytr); dt = time.time() - t
    tr = float(np.sqrt(mean_squared_error(ytr, m.predict(Xtr))))
    print(f"{n:24s} train {tr:.3f}  test {rm(m):.3f}  fit {dt:5.1f}s")

print("\n== Boosting: test RMSE vs #trees (lr=0.1,d=3) and early stopping ==")
g = GradientBoostingRegressor(n_estimators=1500, max_depth=3, learning_rate=.1, random_state=0).fit(Xtr, ytr)
errs = [float(np.sqrt(mean_squared_error(yte, p))) for p in g.staged_predict(Xte)]
print("best n_trees:", int(np.argmin(errs)) + 1, "RMSE", round(min(errs), 3), "| at 1500:", round(errs[-1], 3))

print("\n== Random forest: OOB and max_features ==")
for mf in [0.33, 0.6, 1.0]:
    r = RandomForestRegressor(200, max_features=mf, oob_score=True, n_jobs=2, random_state=0).fit(Xtr, ytr)
    print(f"max_features={mf}: test RMSE {rm(r):.3f}  OOB R2 {r.oob_score_:.3f}")

print("\n== LightGBM leaf-wise overfit on small data (n=300 rows) ==")
Xs, ys = Xtr[:300], ytr[:300]
for nl in [4, 15, 63]:
    m = lgb.LGBMRegressor(n_estimators=300, num_leaves=nl, min_child_samples=5, learning_rate=.1, verbose=-1).fit(Xs, ys)
    print(f"num_leaves={nl:2d}: test RMSE {rm(m):.3f}")

print("\n== Missing values: native handling ==")
Xm = Xte.copy(); rng = np.random.default_rng(0); Xm[rng.random(Xm.shape) < .2] = np.nan
Xtm = Xtr.copy(); Xtm[rng.random(Xtm.shape) < .2] = np.nan
h = HistGradientBoostingRegressor(random_state=0).fit(Xtm, ytr)
print("HistGB trained+tested with 20% NaN: RMSE", round(float(np.sqrt(mean_squared_error(yte, h.predict(Xm)))), 3))

models.clear(); g = None; import gc; gc.collect()
print("\n== Feature importance: impurity vs permutation (RF) ==")
from sklearn.inspection import permutation_importance
r = RandomForestRegressor(100, n_jobs=2, random_state=0).fit(Xtr, ytr)
names = fetch_california_housing().feature_names
pi = permutation_importance(r, Xte, yte, n_repeats=3, random_state=0, n_jobs=2)
for i in np.argsort(-pi.importances_mean)[:4]: print(f"{names[i]:10s} impurity {r.feature_importances_[i]:.3f}  permutation {pi.importances_mean[i]:.3f}")

print("\n== Classification (breast cancer) 5-fold AUC ==")
Xc, yc = load_breast_cancer(return_X_y=True)
for n, m in {"RF": RandomForestClassifier(300, random_state=0), "AdaBoost": AdaBoostClassifier(random_state=0),
             "GBM": GradientBoostingClassifier(random_state=0), "XGB": xgb.XGBClassifier(n_estimators=200, learning_rate=.05, verbosity=0),
             "LGBM": lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, verbose=-1)}.items():
    print(f"{n:9s} {cross_val_score(m, Xc, yc, cv=5, scoring='roc_auc').mean():.3f}")
