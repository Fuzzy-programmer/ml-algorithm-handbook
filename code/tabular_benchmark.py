"""Fair tabular benchmark on OpenML 'adult' (census income, mixed types): same split protocol, preprocessing inside the pipeline."""
import time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer, make_column_selector as sel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb, xgboost as xgb

d = fetch_openml("adult", version=2, as_frame=True); X, y = d.data, (d.target == ">50K").astype(int)
idx = np.random.default_rng(0).choice(len(X), 10000, replace=False); X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
print("rows", len(X), "positive rate", round(y.mean(), 3), "features", X.shape[1], "(numeric:", X.select_dtypes("number").shape[1], ")")
num = sel(dtype_include="number"); cat = sel(dtype_include=["category", "object"])
lin_pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num), ("c", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat)])
tree_pre = ColumnTransformer([("n", SimpleImputer(strategy="median"), num), ("c", make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), cat)])
models = {
 "majority baseline": make_pipeline(lin_pre, DummyClassifier()),
 "Logistic regression": make_pipeline(lin_pre, LogisticRegression(max_iter=2000)),
 "k-NN (k=25)": make_pipeline(lin_pre, KNeighborsClassifier(25)),
 "RBF SVM": make_pipeline(lin_pre, SVC(probability=False)),
 "MLP (64,32)": make_pipeline(lin_pre, MLPClassifier((64, 32), early_stopping=True, max_iter=300, random_state=0)),
 "Random forest (200)": make_pipeline(tree_pre, RandomForestClassifier(200, min_samples_leaf=2, n_jobs=2, random_state=0)),
 "HistGradientBoosting": make_pipeline(tree_pre, HistGradientBoostingClassifier(random_state=0)),
 "XGBoost": make_pipeline(tree_pre, xgb.XGBClassifier(n_estimators=300, learning_rate=.05, max_depth=4, subsample=.8, colsample_bytree=.8, n_jobs=2, verbosity=0)),
 "LightGBM": make_pipeline(tree_pre, lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15, subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=2)),
}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
print(f"{'model':22s} {'ROC-AUC':>14s} {'PR-AUC':>8s} {'logloss':>8s} {'fit s/fold':>10s}")
for n, m in models.items():
    sc = ["roc_auc", "average_precision", "neg_log_loss"] if n != "RBF SVM" and n != "majority baseline" else ["roc_auc", "average_precision"]
    if n == "RBF SVM": m = make_pipeline(lin_pre, SVC())
    try:
        r = cross_validate(m, X, y, cv=cv, scoring=sc if n != "RBF SVM" else ["roc_auc", "average_precision"], n_jobs=1)
        ll = -r["test_neg_log_loss"].mean() if "test_neg_log_loss" in r else float("nan")
        print(f"{n:22s} {r['test_roc_auc'].mean():.3f}±{r['test_roc_auc'].std():.3f}   {r['test_average_precision'].mean():.3f}   {ll:8.3f}   {r['fit_time'].mean():8.2f}")
    except Exception as e: print(n, "ERR", type(e).__name__, str(e)[:80])
