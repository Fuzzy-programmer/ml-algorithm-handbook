# Chapter 6.1 — Time-Series Forecasting: Baselines, ETS, ARIMA/SARIMA, State Space, Boosting, Deep Models

[← Roadmap](../README.md) · Prev [5.7](../05-deep-learning/07-self-supervised.md) · Next [6.2 Recommenders](02-recommender-systems.md) · Priority 🟠 · Code: [`timeseries_demo.py`](../code/timeseries_demo.py)

## 1. Identity & problem
Predict future values of a series ordered in time (demand, load, prices, traffic, sensor values). Data are autocorrelated, non-i.i.d., and often have trend, seasonality, holidays and regime changes. **The evaluation protocol matters as much as the model.**

## 2. History (several lineages, not one)
- **Smoothing:** simple/Holt/Holt–Winters exponential smoothing (Brown 1956; Holt 1957; Winters 1960) → ETS state-space formulation (Hyndman et al. 2002).
- **Stochastic-process models:** AR (Yule 1927), MA (Slutsky/Wold), ARMA; **Box & Jenkins (1970)** ARIMA/SARIMA methodology (identify–estimate–check).
- **State space & filtering:** Kalman filter (1960), structural time series, Prophet (Taylor & Letham 2018, decomposable trend + seasonality + holidays).
- **ML:** lag-feature gradient boosting (LightGBM-based solutions were prominent among top M5 entries), deep learning: DeepAR (Salinas et al. 2017), N-BEATS (Oreshkin 2019), Temporal Fusion Transformer (Lim et al. 2019), PatchTST (Nie 2022), and recent pretrained "foundation" forecasters (e.g. Chronos, TimesFM) — evaluate them cautiously against strong baselines. The M-competitions repeatedly showed simple, well-tuned methods are hard to beat.

## 3. Intuition
A forecast = pattern in the past carried forward: level + trend + repeating seasonal shape + short-term momentum (autocorrelation) + noise.

## 4. Math
- **Stationarity:** constant mean/variance/autocovariance; achieved by differencing (`∇y_t=y_t−y_{t−1}`, seasonal `y_t−y_{t−s}`) or transforms (log). Test with ADF/KPSS.
- **AR(p):** `y_t = c + φ₁y_{t−1}+…+φ_p y_{t−p} + ε_t`. **MA(q):** `y_t = μ + ε_t + θ₁ε_{t−1}+…`. **ARIMA(p,d,q)** = ARMA on d-times differenced data; **SARIMA(p,d,q)(P,D,Q)_s** adds seasonal terms. Identification via ACF/PACF (AR: PACF cuts off; MA: ACF cuts off), then compare by AIC/BIC and residual whiteness (Ljung–Box).
- **Simple exponential smoothing:** `ℓ_t = αy_t + (1−α)ℓ_{t−1}`; forecast = `ℓ_t`. Holt adds trend; Holt–Winters adds seasonality (additive/multiplicative).
**Worked examples.** AR(1) with φ=0.8, c=0, last value 10: forecasts `8, 6.4, 5.12,…` decaying to the mean 0 (mean reversion). SES with α=0.5: observations 10, 12 → `ℓ₁=10`, `ℓ₂=0.5·12+0.5·10=11` (forecast for all horizons).
- **Metrics:** MAE, RMSE, MAPE/sMAPE (fail near 0), **MASE** `= MAE / MAE_in-sample seasonal-naive` (scale-free; <1 beats naive), pinball loss and coverage for probabilistic forecasts (Ch 1.4).
Assumptions: ARIMA — linear dynamics, stationarity after differencing, constant variance; ETS — error/trend/season structure; ML — features carry the information (needs careful lag/leakage handling).

## 5. Training & cost
ARIMA/ETS: per-series fits in milliseconds–seconds (good for many series, "local" models). ML/deep models are usually "global" (one model across many series; shares strength, handles cross-learning, needs more engineering). Multi-step strategies: recursive (feed predictions back), direct (a model per horizon), multi-output.

## 6. When to use / not
| Situation | Prefer |
|---|---|
| Few series, clear trend/seasonality | ETS, SARIMA, Prophet-like decomposition |
| Many related series, covariates, promotions, intermittent demand | Global GBM/NN with lag, calendar and price features |
| Need probabilistic forecasts | Quantile GBM, DeepAR, conformal intervals |
| Very short history | Seasonal naive, ETS; borrow strength via hierarchies/global models |
| Regime shifts / structural breaks | Robust baselines, monitoring, re-fitting, scenario models |
Avoid: shuffled k-fold CV (leaks the future), reporting a single train/test split, forecasting without a naive baseline, using future-known covariates that aren't known at forecast time, deep models on a handful of points.

## 7. Hyperparameters
ARIMA orders (p,d,q)(P,D,Q,s) via `auto_arima`/AIC; ETS error/trend/season type and damping; GBM: lags, rolling stats, calendar features, tree complexity; NN: context length, horizon, patch size, loss; global: normalization per series.

## 8. Implementation
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
sarima = SARIMAX(y, order=(1,1,1), seasonal_order=(0,1,1,12)).fit(disp=False); fc = sarima.forecast(12)
ets = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12).fit(); fc2 = ets.forecast(12)
# Boosted lags: build lag/rolling/calendar features, model the *seasonal difference* (see results), forecast recursively
```

## 9. Evaluation & results (measured)
Data: monthly atmospheric CO₂ (526 months, 1958–2001; statsmodels bundled), clear trend + strong annual seasonality. ADF p-value raw **0.999** (non-stationary) vs **0.0** after first + seasonal differencing. Rolling-origin evaluation: 12-month-ahead forecasts from successive origins in the last 20% of data. MASE uses the in-sample seasonal-naive scale (values > 1 mean worse than the in-sample seasonal-naive error; the test period has trend so even seasonal naive scores 1.27):
| Method | MAE (ppm) | MASE |
|---|---|---|
| Naive (last value) | 1.739 | 1.34 |
| Seasonal naive | 1.647 | 1.27 |
| **Holt–Winters ETS** | 0.472 | 0.36 |
| **SARIMA (1,1,1)(0,1,1,12)** | **0.442** | **0.34** |
| LightGBM on lags, raw target | 1.610 | 1.24 |
| LightGBM on lags, **seasonal-difference** target | 0.822 | 0.63 |
Lessons: (1) classical models win on a single clean series with trend and seasonality; (2) **trees cannot extrapolate a trend** — LightGBM with a raw target ≈ seasonal naive, but modeling the seasonal difference nearly halved the error (1.610 → 0.822); (3) still behind ETS/SARIMA here (one series, 400 training rows) — global ML wins mainly with many series/covariates. ACF of the differenced series at lag 1 and 12: 0.70 and 0.92, confirming remaining seasonality in single-differenced data.

## 10. Case study (template)
Retail demand for thousands of SKUs: baselines (seasonal naive, ETS) → global LightGBM with price/promo/calendar features and per-series scaling → quantile outputs for inventory service levels. Evaluate with rolling-origin CV by week, MASE/WRMSSE, bias, and inventory/business simulation.

## 11. Comparison
| | ETS | ARIMA | Prophet-style | GBM (global) | Deep global |
|---|---|---|---|---|---|
| Few series | Strong | Strong | Good | Weak | Weak |
| Many series + covariates | Weak | Weak | OK | Strong | Strong |
| Interpretability | High | Medium | High | Medium | Low |
| Extrapolates trend | Yes | Yes | Yes | **No (needs differencing/detrending)** | Depends |

## Interview Q&A
- *Why can't you use random k-fold?* It lets future data train the model for past points; use rolling-origin/expanding-window.
- *What does d in ARIMA do?* Number of differences to make the series stationary.
- *ACF vs PACF?* ACF = correlation at lag k including indirect effects; PACF removes intermediate lags — AR order from PACF, MA order from ACF.
- *Why do trees struggle with trends?* They predict averages of seen targets; they can't output values beyond the training range.
- *What is MASE?* MAE scaled by the naive in-sample error.

## Exercises
Fit AR(1) to simulated data and recover φ; build a rolling-origin evaluation with 3 horizons; add Fourier terms and holiday dummies to the boosted model; produce 80% prediction intervals by quantile regression and check coverage.

## Further reading
Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (free, fpp3); Box, Jenkins, Reinsel & Ljung, *Time Series Analysis*; Taylor & Letham (2018); Makridakis et al. M4/M5 papers; Salinas et al. (2020) DeepAR; Oreshkin et al. (2020) N-BEATS; statsmodels/sktime/darts/statsforecast docs.
