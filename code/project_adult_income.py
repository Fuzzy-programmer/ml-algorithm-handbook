"""End-to-end tabular project: baseline -> pipeline -> tuning -> calibration -> threshold -> final test -> error analysis -> model card facts."""
import numpy as np, pandas as pd, warnings, json; warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer, make_column_selector as sel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_score, recall_score, f1_score, log_loss
from sklearn.inspection import permutation_importance
import lightgbm as lgb

d = fetch_openml("adult", version=2, as_frame=True); X, y = d.data, (d.target == ">50K").astype(int)
idx = np.random.default_rng(1).choice(len(X), 20000, replace=False); X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=0)       # final test: touched once
Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=.25, stratify=ytmp, random_state=0)  # validation for calibration/threshold
print("train/val/test", len(Xtr), len(Xva), len(Xte), "| positive rate", round(y.mean(), 3))
num = sel(dtype_include="number"); cat = sel(dtype_include=["category", "object"])
lin_pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num), ("c", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat)])
tree_pre = ColumnTransformer([("n", SimpleImputer(strategy="median"), num), ("c", make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), cat)], verbose_feature_names_out=False)
tree_pre.set_output(transform="pandas")
rep = lambda n, p, yy: print(f"{n:28s} AUC {roc_auc_score(yy,p):.3f}  PR-AUC {average_precision_score(yy,p):.3f}  logloss {log_loss(yy,p):.3f}  Brier {brier_score_loss(yy,p):.4f}")
print("\n-- validation scores --")
base = make_pipeline(lin_pre, LogisticRegression(max_iter=2000)).fit(Xtr, ytr); rep("baseline logistic", base.predict_proba(Xva)[:, 1], yva)
gbm = make_pipeline(tree_pre, lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15, verbose=-1, n_jobs=2)).fit(Xtr, ytr); rep("LightGBM default-ish", gbm.predict_proba(Xva)[:, 1], yva)
space = {"lgbmclassifier__num_leaves": [7, 15, 31, 63], "lgbmclassifier__learning_rate": [.02, .05, .1], "lgbmclassifier__n_estimators": [200, 400, 800],
         "lgbmclassifier__min_child_samples": [10, 20, 50, 100], "lgbmclassifier__subsample": [.6, .8, 1.0], "lgbmclassifier__colsample_bytree": [.5, .8, 1.0], "lgbmclassifier__reg_lambda": [0, 1, 10]}
pipe = make_pipeline(tree_pre, lgb.LGBMClassifier(subsample_freq=1, verbose=-1, n_jobs=2, random_state=0))
rs = RandomizedSearchCV(pipe, space, n_iter=20, scoring="average_precision", cv=StratifiedKFold(4, shuffle=True, random_state=0), random_state=0, n_jobs=1).fit(Xtr, ytr)
print("best params:", {k.split('__')[1]: v for k, v in rs.best_params_.items()}, "| CV PR-AUC", round(rs.best_score_, 3))
best = rs.best_estimator_; p_va = best.predict_proba(Xva)[:, 1]; rep("LightGBM tuned", p_va, yva)
cal = CalibratedClassifierCV(best, method="isotonic", cv=StratifiedKFold(4, shuffle=True, random_state=1)).fit(Xtr, ytr); rep("LightGBM tuned + isotonic", cal.predict_proba(Xva)[:, 1], yva)
pc = cal.predict_proba(Xva)[:, 1]
ths = np.linspace(.1, .8, 36); f1s = [f1_score(yva, pc >= t) for t in ths]; t_best = ths[int(np.argmax(f1s))]
cost = lambda t: 5 * ((pc < t) & (yva == 1)).sum() + 1 * ((pc >= t) & (yva == 0)).sum()     # example: a missed positive costs 5x a false alarm
t_cost = ths[int(np.argmin([cost(t) for t in ths]))]
print(f"F1-optimal threshold {t_best:.2f} (F1 {max(f1s):.3f}) | cost-optimal (FN:FP = 5:1) threshold {t_cost:.2f}")
print("\n-- FINAL TEST (touched once) --")
final = CalibratedClassifierCV(rs.best_estimator_, method="isotonic", cv=StratifiedKFold(4, shuffle=True, random_state=1)).fit(Xtmp, ytmp)
pt = final.predict_proba(Xte)[:, 1]; rep("final model on test", pt, yte)
bt = make_pipeline(lin_pre, LogisticRegression(max_iter=2000)).fit(Xtmp, ytmp); rep("baseline on test", bt.predict_proba(Xte)[:, 1], yte)
for t in [.5, t_best, t_cost]: print(f"thr {t:.2f}: precision {precision_score(yte, pt>=t):.3f} recall {recall_score(yte, pt>=t):.3f} F1 {f1_score(yte, pt>=t):.3f}")
print("\n-- error analysis: AUC by sex and by age band (test) --")
for g, m in [("sex", Xte["sex"]), ("age<30", Xte["age"] < 30), ("age 30-50", (Xte["age"] >= 30) & (Xte["age"] < 50)), ("age>=50", Xte["age"] >= 50)]:
    if g == "sex":
        for v in m.unique(): mm = (m == v).values; print(f"sex={v}: n={mm.sum()} AUC {roc_auc_score(yte[mm], pt[mm]):.3f} pos-rate {yte[mm].mean():.3f} mean-pred {pt[mm].mean():.3f} selection@0.5 {(pt[mm]>=.5).mean():.3f}")
    else: mm = m.values; print(f"{g}: n={mm.sum()} AUC {roc_auc_score(yte[mm], pt[mm]):.3f} pos-rate {yte[mm].mean():.3f} mean-pred {pt[mm].mean():.3f}")
print("\n-- permutation importance (validation PR-AUC drop) --")
pi = permutation_importance(best, Xva, yva, scoring="average_precision", n_repeats=3, random_state=0, n_jobs=1)
for i in np.argsort(-pi.importances_mean)[:6]: print(f"{Xva.columns[i]:16s} {pi.importances_mean[i]:.4f}")
