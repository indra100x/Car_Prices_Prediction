# Car Price Prediction — Multivariate Linear Regression from Scratch

A regression project predicting used car prices from vehicle features, built to practice
the core mechanics behind linear regression: feature engineering, feature scaling, and
gradient descent — all implemented from scratch in NumPy and validated against scikit-learn.

## Project Goal

Predict a car's price from attributes like brand, engine size, mileage, fuel type,
transmission, condition, and age — while implementing the training algorithm (gradient
descent) manually rather than relying on a library's `.fit()` call, in order to understand
what's actually happening under the hood.

## Dataset

`car_price_prediction_.csv` — 2,500 rows, 10 columns:

`Car ID, Brand, Year, Engine Size, Fuel Type, Transmission, Mileage, Condition, Price, Model`

## Workflow

### 1. Data Loading
Loaded the CSV with pandas and did an initial inspection of shape, types, and unique
categorical values.

### 2. Feature Engineering
- **Dropped `Car ID`** — pure index, no predictive value.
- **Dropped `Model`** — too many unique values (Model X, 5 Series, A4, Q7, Mustang, etc.)
  relative to dataset size; one-hot encoding it would have created a large number of
  sparse columns and hurt gradient descent stability. `Brand` was kept as a lower-cardinality
  proxy for the same signal.
- **Converted `Year` → `Car Age`** (`2025 - Year`) so the feature reflects the actual
  relationship with price more directly.
- **One-hot encoded** the remaining categorical columns: `Brand` (7 categories), `Fuel Type`
  (4), `Transmission` (2), `Condition` (3), using `pd.get_dummies(..., drop_first=True)` to
  avoid the dummy variable trap. Boolean output columns were cast to `int`.

Final feature set: 15 columns (numeric + one-hot encoded), target: `Price`.

### 3. Train/Test Split
Split into 80% train / 20% test **before** scaling, to prevent test-set information from
leaking into the training statistics used for scaling.

### 4. Feature Scaling
Three scaling methods were implemented from scratch and compared against scikit-learn's
`StandardScaler`, fit on the training set only and applied to both splits:

| Method | Formula |
|---|---|
| Max Scaling | `x' = x / max(x)` |
| Mean Normalization | `x' = (x - mean(x)) / (max(x) - min(x))` |
| Z-score Normalization | `x' = (x - mean(x)) / std(x)` |
| scikit-learn `StandardScaler` | equivalent to Z-score (reference/validation) |

Validation check: manual Z-score output matched `StandardScaler` output almost exactly
(differences only at the floating-point rounding level), confirming the manual
implementation is correct.

### 5. Cost Function & Gradient — Implemented from Scratch
- `compute_cost(X, w, b, Y)` — Mean Squared Error cost, `J(w,b) = (1/2m) * Σ(f(x) - y)²`
- `compute_gradient(X, w, b, Y)` — partial derivatives `∂J/∂w`, `∂J/∂b`

Both were built first as explicit loop-based versions (to make the underlying math
concrete), then re-implemented as **vectorized versions** using matrix operations
(`X @ w`, `X.T @ errors`) for performance. The two versions were checked against each
other with `np.allclose` to confirm they produce identical results — the vectorized
version is what powers the interactive plots.

Sanity checks performed:
- `compute_cost` at `w=0, b=0` matched the analytical baseline `mean(y²)/2`.
- `compute_gradient`'s `dj_db` at `w=0, b=0` matched the analytical baseline `-mean(y)`.

### 6. Gradient Descent
`gradient_descent(X, Y, w_init, b_init, learning_rate, n_iterations)` iteratively updates
`w` and `b` using the gradient function above and records cost per iteration for
convergence analysis.

### 7. Interactive Learning Rate / Scaling Comparison
An `ipywidgets` slider (`FloatLogSlider`, log scale from 1e-4 to 1) lets you re-run
gradient descent live and compare convergence across the three scaling methods side by
side. Each subplot reports:
- The cost curve over iterations
- Final cost and % cost reduction from the starting point
- Automatic **divergence detection** (flags in red if cost increases or becomes `NaN`)
- Automatic log-scale y-axis when the cost drops by more than 50x, so the flattening
  near convergence stays visible instead of being dwarfed by the initial drop

## Key Findings

- **Z-score normalization and scikit-learn's `StandardScaler` are numerically equivalent**
  by default, and both converge fastest and most reliably across a wide range of learning
  rates.
- **Max scaling** tends to converge more slowly under the same learning rate, since
  dividing by the maximum compresses most values into a narrow band close to 1,
  producing smaller gradient magnitudes per step.
- All three scaling methods are working toward the **same underlying optimum** — what
  differs between them is *how fast* and *how stably* they get there for a given
  learning rate, not the final quality of the solution itself.
- The "best" learning rate is not simply the highest one that still runs — it's the
  one that reaches the lowest cost **within a fixed iteration budget while converging
  smoothly**, without oscillating near the edge of divergence. A learning rate can look
  optimal at 500 iterations but be unstable if pushed further or applied to slightly
  different data.

## Tech Stack

- Python, NumPy, pandas
- Matplotlib (visualization)
- ipywidgets (interactive learning-rate slider)
- scikit-learn (`StandardScaler`, `train_test_split`) — used only for preprocessing and
  as a validation reference, not for training the model itself

## Possible Next Steps

- Extend to Stochastic and Mini-Batch Gradient Descent and compare against full Batch GD
- Evaluate final model on the held-out test set (RMSE, R²) rather than training cost alone
- Add regularization (Ridge/Lasso) to address potential overfitting from the one-hot
  encoded feature space
- Package the trained model behind a simple Streamlit/Flask interface for interactive
  price predictions

## What This Project Demonstrates

This isn't a `model.fit()` exercise — it's linear regression built from the ground up:
cost function, gradient computation, and the optimization loop are all hand-written and
verified against scikit-learn at each step, with an explicit comparison of how feature
scaling choices affect gradient descent's convergence behavior.