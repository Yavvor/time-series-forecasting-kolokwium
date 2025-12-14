# Time Series Forecasting – Kolokwium (Poisson Model)

This repository contains a **time series forecasting framework based on a multiplicative Poisson process decomposition**, developed for the *Statistical Arbitrage Methods* course.

The model decomposes the expected intensity of daily event counts into interpretable components and produces calibrated, noise-aware forecasts.

---

## Model Overview

The time series is modeled as a **Poisson process** with time-varying intensity:

```
λ(t) = trend(t) × weekly(t) × yearly(t) × holiday(t) × spike(t)
```

Where:

- **Trend** – linear regression or moving average  
- **Weekly seasonality** – weekday effects (Monday–Sunday)  
- **Yearly seasonality** – Fourier (sin/cos) or monthly factors  
- **Holiday effects** – automatic detection of low-activity days  
- **Spikes** – detection and controlled amplification of high-demand months  

Residual variability is modeled using **multiplicative noise**, enabling realistic simulations.

---

## Repository Structure

```
.
├── poisson_forecaster.py
├── kolokwium-dane-0912.csv
├── README.md
```

---

## Key Features

- Fully **self-contained Poisson decomposition**
- Automatic **holiday detection**
- **Monthly spike detection** with controlled shrinkage
- **Fourier vs monthly seasonality comparison**
- Hold-out validation and model selection
- Multiple forecast outputs:
  - expected intensity
  - simulated Poisson counts
- Clean diagnostic plots and error metrics

---

## Installation

```bash
pip install numpy pandas matplotlib
```

---

## Usage Example

```python
from poisson_forecaster import KolokwiumForecaster

DATA_PATH = "kolokwium-dane-0912.csv"

f = KolokwiumForecaster(DATA_PATH)

f.load()
f.decompose(
    trend_method="linear",
    yearly_method="fourier",
    holiday_threshold=0.4
)

f.diagnose_components()
f.decomposer.validate_fit()

res = f.evaluate_seasonality_methods(holdout_days=30)
if res["best"] == "monthly":
    f.estimate_monthly_seasonality()

f.forecast_from_decomposer(
    periods=365,
    calibrate=True,
    calibrate_method="mean",
    noise_scale=0.15
)

f.plot_history_and_forecast()
f.save_forecast()
```

---

## Validation Metrics

- **MAE** – Mean Absolute Error  
- **RMSE** – Root Mean Squared Error  
- **MAPE** – Mean Absolute Percentage Error  
- **R²** – Coefficient of Determination  

Metrics are printed automatically after fitting.

---

## Output

The forecast contains:

- `expected_count` – model-implied Poisson intensity  
- `simulated_count` – one simulated realization  
- `lambda` – calibrated intensity parameter  

Saved as:

```
kolokwium_forecast_poisson_365.csv
```

---

## Author

Developed as part of coursework for **Statistical Arbitrage Methods**.
