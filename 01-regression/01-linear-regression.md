# Chapter 1.1 — Linear Regression (Ordinary Least Squares)

[← Roadmap](../README.md) · Next: [1.2 Polynomial regression & basis expansion](02-polynomial-regression.md)

| | |
|---|---|
| **Family** | Supervised → Regression → Linear models |
| **Task** | Predict a continuous target; also explain relationships |
| **Priority** | 🔴 MUST — the baseline for every regression problem |
| **Code** | [`code/linear_regression_scratch.py`](../code/linear_regression_scratch.py) · [`code/linear_regression_sklearn.py`](../code/linear_regression_sklearn.py) |

---

## 1. Identity
Linear regression models the target as a weighted sum of features plus noise, and chooses the weights that minimize the **sum of squared errors** (OLS = Ordinary Least Squares). With one feature it is *simple* linear regression; with several, *multiple* linear regression.

## 2. Problem statement
*"How much does a house cost, given its size, location and income of the neighborhood?"* More generally: given numeric inputs `x`, predict a number `y`, and learn how much each input moves `y` when the others are held fixed.

## 3. Historical background
- **Before:** astronomers and surveyors had more measurements than unknowns and noisy data. Ad-hoc approaches (averaging, picking the "best" subset of equations) gave inconsistent answers.
- **1805:** Adrien-Marie Legendre published the method of least squares. **Gauss** claimed to have used it since ~1795 and in 1809 tied it to the normal error distribution — a priority dispute (see Stigler 1981).
- **1886:** Francis Galton coined "regression" (regression *toward the mean* in heights of parents and children); Pearson and Yule formalized correlation/multiple regression around 1890–1900; Fisher built the inference machinery (ANOVA, F-tests) in the 1920s.
- **Gauss–Markov theorem** (Gauss 1821; Markov early 1900s): under mild assumptions OLS is the *Best Linear Unbiased Estimator* — no assumption of normality needed for that result.
- **Why it still matters:** it is the root of Ridge/Lasso (Chapter 1.3), logistic regression and GLMs (Chapter 2.1), and even a single neural-network layer.

## 4. Core intuition
Draw the straight line through a scatter of points so that the vertical gaps (residuals) are, in total, as small as possible. Squaring the gaps (a) makes positive and negative errors count equally, (b) punishes big misses heavily, (c) makes the math smooth and solvable in closed form.

```
 y │            ·
   │         ·  /      ← vertical distance from each point to the line
   │      ·   /·          = residual  eᵢ = yᵢ − ŷᵢ
   │    · /·             OLS picks the line minimizing  Σ eᵢ²
   │   ·/
   │  /·
   └───────────────── x
```
Geometric view: `y` is a vector in n-dimensional space; the columns of `X` span a plane (subspace). OLS **projects `y` orthogonally onto that plane** — the prediction `ŷ` is the closest point in the plane, and the residual is perpendicular to every feature.

## 5. Mathematics

### 5.1 Model
$$y_i = \beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip} + \varepsilon_i \quad\Longleftrightarrow\quad \mathbf{y} = X\boldsymbol\beta + \boldsymbol\varepsilon$$
(`X` includes a column of ones for the intercept.)

### 5.2 Loss
$$J(\boldsymbol\beta) = \lVert \mathbf{y}-X\boldsymbol\beta\rVert_2^2 = \sum_i (y_i-\hat y_i)^2$$

### 5.3 Closed-form solution
Set the gradient to zero: $\nabla_\beta J = -2X^\top(\mathbf y - X\boldsymbol\beta)=0$, giving the **normal equations**
$$X^\top X\,\hat{\boldsymbol\beta} = X^\top \mathbf y \quad\Rightarrow\quad \hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top\mathbf y .$$
The condition $X^\top(\mathbf y - X\hat\beta)=0$ is exactly "residuals are orthogonal to every feature". $J$ is convex, so this stationary point is the global minimum (unique if $X$ has full column rank).

For one feature: $\hat\beta_1 = \dfrac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2} = \dfrac{S_{xy}}{S_{xx}}$, $\;\hat\beta_0=\bar y-\hat\beta_1\bar x$.

### 5.4 Probabilistic view (MLE)
If $\varepsilon_i\sim\mathcal N(0,\sigma^2)$ independently, the likelihood of the data is Gaussian and **maximizing it is identical to minimizing the sum of squares**. Squared error is therefore not arbitrary: it is the right loss when noise is Gaussian. (Laplace noise → absolute error → quantile/median regression.)

### 5.5 Assumptions (and what each one buys you)
| # | Assumption | Needed for | Check with |
|---|---|---|---|
| 1 | **Linearity** in parameters: $E[y\mid x]$ is linear in the (transformed) features | Unbiased predictions | Residual vs. fitted plot (no curve) |
| 2 | **Exogeneity** $E[\varepsilon\mid X]=0$ (no omitted confounders, no leakage) | Unbiased coefficients | Domain reasoning (not testable from data alone) |
| 3 | **No perfect collinearity** | Unique solution | VIF, condition number of X |
| 4 | **Homoscedasticity** (constant error variance) | Correct standard errors; "best" in Gauss–Markov | Residual spread plot, Breusch–Pagan |
| 5 | **Uncorrelated errors** | Correct standard errors | Durbin–Watson, residual ACF (time data) |
| 6 | **Normal errors** | Exact t/F tests and CIs in small samples | Q–Q plot of residuals |

For *pure prediction* only (1) matters much. For *inference on coefficients* (2)–(6) matter. Gauss–Markov (unbiased + minimum variance among linear unbiased estimators) needs (1)–(5), not normality.

### 5.6 Worked example (by hand)
Data: $x=[1,2,3,4,5]$, $y=[2,4,5,4,5]$.

**Slope/intercept:** $\bar x=3,\ \bar y=4$.
$S_{xy}=(-2)(-2)+(-1)(0)+0(1)+1(0)+2(1)=6,\quad S_{xx}=4+1+0+1+4=10$
$\hat\beta_1=6/10=0.6,\qquad \hat\beta_0=4-0.6\cdot3=2.2$

**Same result via normal equations.** With rows $[1,x_i]$:
$X^\top X=\begin{bmatrix}5&15\\15&55\end{bmatrix},\; X^\top y=\begin{bmatrix}20\\66\end{bmatrix},\; (X^\top X)^{-1}=\tfrac1{50}\begin{bmatrix}55&-15\\-15&5\end{bmatrix}$
$\hat\beta=\tfrac1{50}\begin{bmatrix}55\cdot20-15\cdot66\\-15\cdot20+5\cdot66\end{bmatrix}=\tfrac1{50}\begin{bmatrix}110\\30\end{bmatrix}=\begin{bmatrix}2.2\\0.6\end{bmatrix}$ ✓

**Fit quality:** $\hat y=[2.8,3.4,4.0,4.6,5.2]$, residuals $[-0.8,\,0.6,\,1.0,\,-0.6,\,-0.2]$ (they sum to 0, as they must with an intercept).
$SSE=0.64+0.36+1+0.36+0.04=2.4,\quad SST=\sum(y-\bar y)^2=6$
$R^2=1-SSE/SST=1-2.4/6=\mathbf{0.60}$

**Uncertainty:** $\hat\sigma^2=SSE/(n-p-1)=2.4/3=0.8$; $SE(\hat\beta_1)=\sqrt{\hat\sigma^2/S_{xx}}=\sqrt{0.08}=0.283$; $t=0.6/0.283=2.12$ on 3 d.f. → two-sided $p\approx0.12$; 95% CI $0.6\pm3.182\cdot0.283=[-0.30,\,1.50]$.
*Reading:* with only 5 points the slope is **not** statistically distinguishable from zero even though $R^2=0.6$. Small samples are noisy.

**One gradient-descent step** (starting at $\beta=0$, lr = 0.05): gradient $=\tfrac{2}{n}X^\top(X\beta-y)=-0.4\,[20,\,66]=[-8,\,-26.4]$, so $\beta\leftarrow[0.4,\;1.32]$. The slope overshoots its final value 0.6 — a hint why feature scaling and learning rate matter (Section 6).

### 5.7 Useful facts
- **R²** = fraction of variance explained *on that data*; never decreases when you add features → use adjusted R² or held-out data.
- **Variance of coefficients:** $\mathrm{Var}(\hat\beta)=\sigma^2(X^\top X)^{-1}$. Near-collinear columns make $(X^\top X)^{-1}$ huge.
- **Variance Inflation Factor:** $\mathrm{VIF}_j = 1/(1-R_j^2)$ where $R_j^2$ comes from regressing feature $j$ on the others; VIF > 5–10 is a common warning level.

## 6. Training and prediction
| Solver | How | Cost | Use when |
|---|---|---|---|
| Normal equation | solve $X^\top X\beta=X^\top y$ | $O(np^2+p^3)$ | Teaching; small p. Squares the condition number → numerically fragile |
| QR / SVD (`np.linalg.lstsq`, sklearn default) | factor $X$ directly | $O(np^2)$ | **Default** — stable, handles rank deficiency (min-norm solution) |
| Gradient descent / SGD | iterate $\beta\leftarrow\beta-\eta\nabla J$ | $O(np)$ per epoch | Huge n or p, streaming, or when you will add regularization/online updates |
| Sparse iterative (LSQR, CG) | Krylov methods | $O(\text{nnz})$/iter | Large sparse X |

- **Prediction:** $\hat y=x^\top\hat\beta$, cost $O(p)$ per row, memory $O(p)$ — one of the cheapest models to serve.
- **GD stopping:** gradient norm below tolerance, loss change below tolerance, or max iterations. Convexity guarantees no local minima, but badly scaled features give a long, narrow valley → slow or divergent GD. **Standardize features** (shown in the code).

## 7. Evolution and advancements
Linear regression is a *root*, so several lines descend from it for different reasons — it's a tree, not a single timeline:

| Descendant | Problem with OLS it addresses | What it adds | Cost |
|---|---|---|---|
| Polynomial / basis expansion (1.2) | Relationship is curved | Nonlinear in $x$, still linear in $\beta$ | Variance explodes at high degree |
| Ridge (Hoerl & Kennard 1970) (1.3) | Collinearity, overfitting | L2 penalty shrinks coefficients | Biased; needs λ |
| Lasso (Tibshirani 1996) (1.3) | Too many features | L1 → exact zeros (selection) | Unstable with correlated features |
| Elastic Net (Zou & Hastie 2005) (1.3) | Lasso instability | L1+L2 | Two hyperparameters |
| Quantile regression (Koenker & Bassett 1978) (1.4) | Want median/percentiles, outlier-robust | Pinball loss | No closed form |
| Huber / RANSAC / Theil–Sen | Outliers | Robust loss | Slower, extra knobs |
| WLS / GLS | Heteroscedastic or correlated errors | Weighted fits | Need the error structure |
| GLMs: logistic, Poisson (Nelder & Wedderburn 1972) (2.1) | Target is binary/count | Link function + non-Gaussian likelihood | Iterative fitting (IRLS) |
| GAMs | Unknown nonlinear shapes per feature | Additive smooth functions | More tuning, still additive |
| Neural nets (Phase 4) | Complex interactions | Learned nonlinear features | Data, compute, opacity |

**Not a strict upgrade path:** on small/clean/near-linear data, plain OLS or Ridge often beats gradient boosting and is far easier to explain.

## 8. When to use
- A strong, **interpretable baseline** for any numeric target (always fit it first).
- Relationship is roughly linear after sensible transforms; n much larger than p.
- You need coefficients with confidence intervals (pricing, econometrics, A/B-test adjustment, scientific work).
- Tight latency/memory budgets; needs to be auditable.
- Sparse high-dimensional data with regularization (Ridge/Lasso variants).

## 9. When not to use
| Situation | Why it fails | Better option |
|---|---|---|
| Strong nonlinearity / interactions | High bias | Add features/splines; boosted trees; NN |
| p ≥ n or strongly collinear features | Non-unique/unstable $\beta$ | Ridge, Lasso, PCA regression |
| Heavy outliers | Squared loss is dominated by them | Huber, quantile, RANSAC |
| Binary / count / positive-skew target | Predictions outside valid range; wrong noise model | Logistic, Poisson/Gamma GLM |
| Autocorrelated errors (time series) | Invalid SEs, leakage-prone CV | ARIMA/ETS, GLS, time-aware CV |
| Extrapolating far beyond training range | Line keeps going | Domain model, saturating forms |
| Causal claims from observational data | Confounding breaks exogeneity | Causal methods (Phase 5) |

## 10. Strengths and weaknesses
**Strengths:** closed-form optimum, convex (no local minima), very fast, tiny model, interpretable, well-developed inference theory, extends naturally (regularization, GLMs, basis expansions).
**Weaknesses:** sensitive to outliers and leverage points, assumes linear additive structure, unstable under collinearity, can't learn interactions automatically, scale of coefficients depends on feature units (standardize to compare).

## 11. Hyperparameters
Plain OLS has essentially none — one reason it's a clean baseline. The knobs live in the surrounding choices:

| Knob | Effect |
|---|---|
| `fit_intercept` | Include $\beta_0$. Turn off only if data is centered or truth passes through the origin |
| Feature scaling | Doesn't change OLS predictions, but **does** change GD convergence and any regularized variant |
| Feature transforms (log, polynomial, interactions) | Where the real modeling happens |
| Target transform (e.g. log y) | Fixes skew/heteroscedasticity; back-transform bias (see case study) |
| GD: learning rate, iterations | Too large diverges (demo in code); too small is slow |
| `positive=True` (sklearn) | Non-negative coefficients |
| Regularization λ (Ridge/Lasso) | Variance↓, bias↑ → Chapter 1.3 |

## 12. Implementation

### 12.1 From scratch (NumPy) — `code/linear_regression_scratch.py`
```python
import numpy as np

class LinearRegressionScratch:
    def __init__(self, method="lstsq", lr=0.05, n_iter=5000, fit_intercept=True):
        self.method, self.lr, self.n_iter, self.fit_intercept = method, lr, n_iter, fit_intercept

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return np.c_[np.ones(len(X)), X] if self.fit_intercept else X

    def fit(self, X, y):
        A, y = self._design(X), np.asarray(y, dtype=float)
        if self.method == "normal":      # (AᵀA)⁻¹Aᵀy  -- textbook, numerically fragile
            w = np.linalg.solve(A.T @ A, A.T @ y)
        elif self.method == "lstsq":     # SVD-based, what you should use
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
        elif self.method == "gd":        # batch gradient descent on MSE
            w = np.zeros(A.shape[1])
            for _ in range(self.n_iter):
                w -= self.lr * (2 / len(y)) * A.T @ (A @ w - y)
        if self.fit_intercept: self.intercept_, self.coef_ = w[0], w[1:]
        else:                  self.intercept_, self.coef_ = 0.0, w
        return self

    def predict(self, X):
        return self._design(X) @ np.r_[self.intercept_ if self.fit_intercept else [], self.coef_]

    def score(self, X, y):               # R²
        y = np.asarray(y, dtype=float)
        return 1 - ((y - self.predict(X)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
```
Actual output on the worked example (all three solvers agree with Section 5.6):
```
normal  b0=2.2000 b1=0.6000 R2=0.6000
lstsq   b0=2.2000 b1=0.6000 R2=0.6000
gd      b0=2.2000 b1=0.6000 R2=0.6000
lr=0.1 unscaled -> b1 = -6.7e+06 (diverged)
lr=0.1 scaled   -> b0=4.0000 b1=0.8485 (converged)
```
Note the last two lines: the *same* learning rate diverges on raw `x` (values up to 5) but converges once `x` is standardized. (Scaled coefficients are in standard-deviation units, so they differ from 0.6 but describe the same line.)

### 12.2 Library (scikit-learn) — `code/linear_regression_sklearn.py`
```python
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv = KFold(5, shuffle=True, random_state=0)
model = make_pipeline(StandardScaler(), LinearRegression())   # scaler lives INSIDE the pipeline: no leakage
scores = cross_validate(model, X_train, y_train, cv=cv,
                        scoring=["r2", "neg_root_mean_squared_error"])
```
Use `statsmodels.api.OLS` when you need p-values, confidence intervals and diagnostics (`summary()`); use scikit-learn when you need a pipeline component.

## 13. Evaluation
| Metric | Formula | Notes |
|---|---|---|
| RMSE | $\sqrt{\tfrac1n\sum e_i^2}$ | Same units as $y$; penalizes large errors |
| MAE | $\tfrac1n\sum \lvert e_i\rvert$ | Robust, easy to explain |
| R² | $1-SSE/SST$ | 0 = no better than predicting the mean; can be negative on test data |
| MAPE / sMAPE | relative error | Breaks when $y\approx0$ |

**Protocol:** (1) hold out a test set you touch once; (2) K-fold CV on the training part for model selection; (3) always report a **baseline** (predict the mean); (4) keep scaling/imputation inside the pipeline; (5) for time-ordered or grouped data use `TimeSeriesSplit` / `GroupKFold`.

**Residual diagnostics:** residual-vs-fitted (curvature → nonlinearity; funnel → heteroscedasticity), Q–Q plot (normality), residuals vs. each feature, leverage/Cook's distance (influential points), VIF (collinearity). **Overfitting check:** train vs. CV error gap (small for OLS unless p is large relative to n).

## 14. Practical case study — California housing
Target: median house value (units of $100k) for 20,640 census block groups; 8 numeric features. 80/20 split, 5-fold CV on the training part. Results from running `linear_regression_sklearn.py`:

| Model | CV R² | CV RMSE | CV MAE |
|---|---|---|---|
| Baseline (predict mean) | −0.000 | 1.156 | 0.914 |
| **Linear regression** | **0.612** | **0.720** | **0.529** |
| Linear on log(y), back-transformed | 0.310 | 0.958 | 0.548 |

Final test set (touched once): **R² 0.576, RMSE 0.746, MAE 0.533** (≈ $75k typical error).

Standardized coefficients (largest first): Latitude −0.897, Longitude −0.870, MedInc +0.854, AveBedrms +0.339, AveRooms −0.294, HouseAge +0.123, AveOccup −0.041, Population −0.002.

**Interpretation and caveats**
- A plain line cuts RMSE by ~38% versus the baseline: a real signal, but R² ≈ 0.6 means a lot is unexplained.
- Income (MedInc) is the dominant interpretable driver. Latitude/longitude have large coefficients because location matters nonlinearly (coast, cities); a *line* in lat/long can only approximate that → boosted trees or spatial features should do much better (compare in Chapter 3.4).
- AveRooms is negative while AveBedrms is positive: these two are correlated, so individual signs are not trustworthy — the classic collinearity symptom. Don't read causal meaning into them.
- The log-target variant did *worse* in original units. Hypothesis (not tested here): the dataset's target is capped at 5.0, and exponentiating predictions amplifies errors at the high end. The lesson: a transform that "looks statistically proper" must be judged on the metric you care about, in original units.
- CV R² 0.612 vs. test 0.576: a gap of this size is plausible sampling variation between folds and a single test split, not evidence of overfitting for a 9-parameter model on ~16k rows.

**Collinearity demo (simulation, n=100, 300 repeats, true β₁=1):** std of the estimated $\hat\beta_1$ is 0.10 when features are uncorrelated and **0.75 at correlation 0.99** — predictions stay fine, but individual coefficients become unreliable.

## 15. Comparison and decision guide
| vs. | Choose OLS when | Choose the other when |
|---|---|---|
| Ridge/Lasso/ElasticNet | n ≫ p, little collinearity, want classical inference | Collinearity, p large, want selection or best predictive accuracy |
| Polynomial/splines | Roughly linear | Known curvature in few features |
| Random Forest | Need coefficients, extrapolation, speed, small data | Interactions and nonlinearity dominate |
| Gradient boosting | Explainability/regulation, baseline | Maximum tabular accuracy |
| Neural net | Small tabular data | Unstructured data or huge datasets |
| Logistic/Poisson GLM | Target continuous, roughly Gaussian noise | Binary/count/skewed targets |

**Decision flow:** fit OLS → check residual plots → curved? add transforms/splines · collinear or p large? Ridge/Lasso · outliers? Huber/quantile · wrong target type? GLM · still unexplained structure? trees/boosting. Keep OLS as the benchmark all fancier models must beat.

## 16. Further reading
- Legendre (1805), *Nouvelles méthodes pour la détermination des orbites des comètes*; Gauss (1809), *Theoria motus corporum coelestium*.
- S. Stigler (1981), "Gauss and the Invention of Least Squares," *Annals of Statistics* 9(3).
- Hoerl & Kennard (1970), "Ridge Regression," *Technometrics*; Tibshirani (1996), "Regression Shrinkage and Selection via the Lasso," *JRSS B*.
- ISLR (James et al.), Ch. 3; ESL (Hastie et al.), Ch. 3 — both free from the authors.
- scikit-learn: *Linear Models* user-guide page; `statsmodels` OLS docs.
- Open questions / practice issues: valid inference after model selection; high-dimensional inference; robustness to distribution shift.

---

## Interview Q&A
**30-second answer — "Explain linear regression."** It fits a hyperplane that minimizes the sum of squared residuals. It has a closed-form solution $(X^\top X)^{-1}X^\top y$, assumes a linear, additive relationship with roughly constant-variance independent errors, and is the interpretable baseline for any numeric target.

**Follow-ups**
1. *Why squared error?* Smooth and convex → closed form; it's the MLE under Gaussian noise; trade-off: sensitive to outliers.
2. *What if $X^\top X$ is singular?* Perfect collinearity or p > n; use Ridge, drop/merge features, or the SVD min-norm solution.
3. *Do you need to scale features?* Not for OLS predictions; yes for GD, regularization, and comparing coefficient sizes.
4. *R² improved after adding a feature — good?* Not necessarily; R² never decreases on training data. Check adjusted R² or CV.
5. *Which assumptions can you check?* Linearity, homoscedasticity, normality, autocorrelation via residual plots/tests; exogeneity only via domain reasoning.
6. *Closed form vs. GD at scale?* Closed form is $O(np^2)$; with huge n/p or streaming data use SGD or sparse iterative solvers.
7. *Coefficient signs look wrong.* Suspect collinearity; check VIF, try Ridge, don't over-interpret.

## Exercises
1. Re-derive $\hat\beta_1=S_{xy}/S_{xx}$ by setting $\partial J/\partial\beta_0=\partial J/\partial\beta_1=0$.
2. Add the point (6, 20) to the worked example. How do slope, intercept and R² change? (Leverage and influence.)
3. Run the scratch GD with learning rates 0.001, 0.01, 0.05, 0.1 on raw vs. standardized `x`; plot loss per iteration.
4. Using `statsmodels`, fit California housing and read the `summary()`: which coefficients have wide CIs? Compare with VIFs.
5. Fit Ridge with `RidgeCV` on the same split. Does CV RMSE improve over 0.720? Why or why not?
6. Add `Latitude × Longitude` and squared terms. How far does CV RMSE move, and what does it tell you about the earlier "location is nonlinear" claim?
