"""Time series: baselines, ETS, ARIMA/SARIMA, boosted lag features; rolling-origin evaluation on CO2 data (statsmodels bundled)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf
import lightgbm as lgb
y = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
print("monthly CO2 points:", len(y), y.index[0].date(), "->", y.index[-1].date())
print("ADF p-value raw:", round(adfuller(y)[1], 3), "| after d=1 & seasonal diff:", round(adfuller(y.diff().diff(12).dropna())[1], 4))
H, split = 12, int(len(y) * .8)
mase_den = np.mean(np.abs(y.values[12:] - y.values[:-12]))             # seasonal-naive in-sample scale
def evaluate(fn, origins=range(split, len(y) - H, 12)):
    errs = []
    for o in origins:
        tr, te = y.iloc[:o], y.iloc[o:o + H]; errs.append(np.mean(np.abs(te.values - fn(tr, H))))
    return np.mean(errs) / mase_den, np.mean(errs)
naive = lambda tr, h: np.repeat(tr.values[-1], h)
snaive = lambda tr, h: tr.values[-12:][:h]
def ets(tr, h): return ExponentialSmoothing(tr, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(h).values
def sarima(tr, h): return SARIMAX(tr, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).forecast(h).values
def gbm_factory(diff):
    def gbm(tr, h):
        d = pd.DataFrame({"y": tr.values})
        for l in [1, 2, 3, 6, 12, 24]: d[f"l{l}"] = d.y.shift(l)
        d["m"] = tr.index.month; d = d.dropna()
        target = d.y - d.l12 if diff else d.y          # diff=True: model the seasonal difference so the trend is not extrapolated by trees
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=.05, num_leaves=8, verbose=-1).fit(d.drop(columns="y"), target)
        hist = list(tr.values); out = []
        for k in range(h):
            row = {f"l{l}": hist[-l] for l in [1, 2, 3, 6, 12, 24]}; row["m"] = (tr.index[-1].month + k) % 12 + 1
            p = m.predict(pd.DataFrame([row]))[0]; p = p + row["l12"] if diff else p; out.append(p); hist.append(p)
        return np.array(out)
    return gbm
print("\n12-month-ahead rolling-origin results (MAE in ppm; MASE vs seasonal naive scale):")
for n, f in [("naive (last value)", naive), ("seasonal naive", snaive), ("Holt-Winters ETS", ets), ("SARIMA(1,1,1)(0,1,1,12)", sarima), ("LightGBM lags, raw target", gbm_factory(False)), ("LightGBM lags, seasonal-diff target", gbm_factory(True))]:
    mase, mae = evaluate(f); print(f"{n:28s} MAE {mae:6.3f}  MASE {mase:5.2f}")
print("\nACF lags 1,12 of the differenced series:", acf(y.diff().dropna(), nlags=12)[[1, 12]].round(2))
