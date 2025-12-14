# Standard library import for filesystem path handling
import os

# Typing helpers for clearer type annotations
from typing import Optional, Dict, Any

# Data analysis and numerical libraries
import pandas as pd
import numpy as np

# Plotting library for diagnostics and visualization
import matplotlib.pyplot as plt


class PoissonProcessDecomposer:
    """
    Lightweight in-file decomposer:
      λ(t) = trend(t) × week_pattern(t) × year_pattern(t) × holiday_effect(t)

    This class decomposes a time series assumed to follow a Poisson process
    into multiplicative components representing trend, weekly seasonality,
    yearly seasonality, holidays, and spikes. It also estimates residual noise
    to enable realistic forecasts.
    """

    def __init__(self, df: pd.DataFrame, date_col: str = "date", value_col: str = "calls"):
        # Store a copy of the input data
        self.df = df.copy()

        # Column names for date and observed counts
        self.date_col = date_col
        self.value_col = value_col

        # Ensure date column is datetime
        self.df[date_col] = pd.to_datetime(self.df[date_col])

        # Sort data chronologically
        self.df = self.df.sort_values(date_col).reset_index(drop=True)

        # Precompute calendar-based features
        self.df["day_of_week"] = self.df[date_col].dt.dayofweek
        self.df["day_of_year"] = self.df[date_col].dt.dayofyear
        self.df["month"] = self.df[date_col].dt.month

        # Time index used for trend estimation
        self.df["time_index"] = np.arange(len(self.df))

        # Containers for model components and parameters
        self.components: Dict[str, np.ndarray] = {}
        self.weekly_params: Dict[int, float] = {}
        self.yearly_params: Dict[Any, Any] = {}
        self.holiday_params: Dict[str, Any] = {}
        self.spike_params: Dict[str, Any] = {}

        # Final estimated intensity λ(t)
        self.lambda_estimated: Optional[np.ndarray] = None

        # Parameters of the trend model
        self.trend_params: Optional[np.ndarray] = None

        # Coefficient of variation for multiplicative residual noise
        self.residual_cv: float = 0.1  # fallback default

    def estimate_trend(self, method: str = "linear"):
        # Estimate long-term trend component
        if method == "linear":
            # Linear regression on time index
            X = np.vstack([self.df["time_index"].values, np.ones(len(self.df))]).T
            y = self.df[self.value_col].values
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.trend_params = coef
            trend = X @ coef
        elif method == "moving_average":
            # Rolling mean as a non-parametric trend
            trend = self.df[self.value_col].rolling(
                window=7, center=True, min_periods=1
            ).mean().values
            self.trend_params = None
        else:
            raise ValueError("Unknown trend method")

        # Enforce positivity
        self.components["trend"] = np.maximum(trend, 1e-6)
        return self.components["trend"]

    def estimate_weekly_seasonality(self, weekly_strength: float = 1.0):
        """
        Estimate weekly multiplicative factors.
        weekly_strength controls amplification of weekday/weekend contrast.
        """
        # Remove trend from data
        detrended = self.df[self.value_col] / (self.components.get("trend", 1.0))

        # Average by weekday
        weekly_means = detrended.groupby(self.df["day_of_week"]).mean()
        overall = weekly_means.mean()

        # Normalize so mean is 1
        weekly_norm = (weekly_means / (overall + 1e-12))

        # Amplify deviations from 1 if requested
        adjusted = 1.0 + weekly_strength * (weekly_norm - 1.0)

        # Clamp values to avoid extreme multipliers
        adjusted_clamped = adjusted.clip(lower=0.7, upper=1.6)

        # Store weekday → multiplier mapping
        self.weekly_params = {
            int(k): float(adjusted_clamped.loc[k]) for k in adjusted_clamped.index
        }

        # Assign weekly pattern to each observation
        self.components["week_pattern"] = (
            self.df["day_of_week"].map(self.weekly_params).values
        )
        return self.components["week_pattern"]

    def estimate_yearly_seasonality(self, method: str = "fourier"):
        # Copy observed values for detrending
        detrended = self.df[self.value_col].copy().astype(float)

        # Remove already-estimated components
        if "trend" in self.components:
            detrended = detrended / (self.components["trend"] + 1e-12)
        if "week_pattern" in self.components:
            detrended = detrended / (self.components["week_pattern"] + 1e-12)

        if method == "fourier":
            # Fourier series with one harmonic for annual seasonality
            doy = self.df["day_of_year"].values
            X = np.column_stack([
                np.ones_like(doy),
                np.sin(2 * np.pi * doy / 365),
                np.cos(2 * np.pi * doy / 365)
            ])
            coeffs, *_ = np.linalg.lstsq(X, detrended.values, rcond=None)
            year_raw = X @ coeffs

            # Normalize to mean 1
            year_norm = year_raw / (year_raw.mean() + 1e-12)

            # Store parameters and component
            self.yearly_params = {"coefficients": coeffs}
            self.components["year_pattern"] = year_norm

        elif method == "monthly":
            # Discrete monthly seasonality
            monthly_means = detrended.groupby(self.df["month"]).mean()
            overall = monthly_means.mean()
            month_map = (monthly_means / (overall + 1e-12)).to_dict()

            # Store month → multiplier mapping
            self.yearly_params = {int(k): float(v) for k, v in month_map.items()}
            self.components["year_pattern"] = (
                self.df["month"].map(self.yearly_params).values
            )
        else:
            raise ValueError("Unknown yearly method")

        return self.components["year_pattern"]

    def detect_holidays(self, threshold: float = 0.5,
                        spike_threshold: float = 1.4,
                        spike_scale: float = 0.5):
        """
        Detect low-demand holidays and high-demand monthly spikes.
        Holidays are days with unusually low residuals.
        Spikes are months with unusually high residual maxima.
        """
        # Start from observed values
        residual = self.df[self.value_col].astype(float).copy()

        # Remove all estimated components
        for name in ("trend", "week_pattern", "year_pattern"):
            if name in self.components:
                residual = residual / (self.components[name] + 1e-12)

        # Holiday detection via low residuals
        median = residual.median()
        is_holiday = residual < (median * threshold)

        # Apply fixed multiplicative reduction on holidays
        holiday_effect = np.where(is_holiday, 0.3, 1.0)
        self.components["holiday_effect"] = holiday_effect

        # Store holiday metadata
        self.holiday_params = {
            "dates": self.df.loc[is_holiday, self.date_col].dt.date.tolist(),
            "count": int(is_holiday.sum()),
            "effect": 0.3
        }

        # Detect upward monthly spikes
        dfm = self.df.assign(resid=residual.values)
        monthly_max = dfm.groupby("month")["resid"].max()
        monthly_med = dfm.groupby("month")["resid"].median().replace(0, 1e-12)
        month_ratio = (monthly_max / monthly_med).fillna(1.0)

        # Raw spike multipliers
        raw_map = {
            int(m): float(month_ratio.loc[m])
            if month_ratio.loc[m] > spike_threshold else 1.0
            for m in month_ratio.index
        }

        # Shrink spikes toward 1.0
        shrunk = {
            m: 1.0 + spike_scale * (raw_map[m] - 1.0) for m in raw_map
        }

        # Clamp extreme values
        shrunk_clamped = {
            m: min(3.5, max(0.85, shrunk[m])) for m in shrunk
        }

        # Store spike parameters
        self.spike_params = {
            "raw_map": raw_map,
            "month_map": shrunk_clamped,
            "threshold": spike_threshold,
            "scale": spike_scale
        }

        return holiday_effect

    def estimate_residual_noise(self):
        """
        Estimate coefficient of variation of multiplicative residuals.
        """
        # Build lambda if not already estimated
        if self.lambda_estimated is None:
            lam = np.ones(len(self.df))
            for comp in self.components.values():
                lam = lam * comp
        else:
            lam = np.asarray(self.lambda_estimated)

        # Ratio of observed counts to fitted lambda
        ratio = self.df[self.value_col].values / (lam + 1e-12)

        # CV = std / mean
        cv = float(np.std(ratio) / (np.mean(ratio) + 1e-12))

        # Enforce a minimum variance
        self.residual_cv = max(0.05, cv)
        return self.residual_cv

    def estimate_lambda(self):
        # Combine all multiplicative components into final intensity
        lam = np.ones(len(self.df))
        for comp in self.components.values():
            lam = lam * comp

        # Store results
        self.lambda_estimated = lam
        self.df["lambda_estimated"] = lam
        return lam

    def fit(self, trend_method: str = "linear",
            yearly_method: str = "fourier",
            holiday_threshold: float = 0.5,
            weekly_strength: float = 1.0,
            spike_threshold: float = 1.4,
            spike_scale: float = 0.5):
        # Full estimation pipeline
        self.estimate_trend(method=trend_method)
        self.estimate_weekly_seasonality(weekly_strength=weekly_strength)
        self.estimate_yearly_seasonality(method=yearly_method)
        self.detect_holidays(threshold=holiday_threshold,
                             spike_threshold=spike_threshold,
                             spike_scale=spike_scale)
        self.estimate_lambda()
        self.estimate_residual_noise()
        return self

    def validate_fit(self) -> Dict[str, float]:
        """
        Validate model fit using standard error metrics:
        MAE, RMSE, MAPE, and R².
        """
        actual = self.df[self.value_col].values
        predicted = self.lambda_estimated

        # Error metrics
        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mape = float(
            np.mean(np.abs((actual - predicted) / (actual + 1e-12)))
        ) * 100

        # Coefficient of determination
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = float(1 - (ss_res / (ss_tot + 1e-12)))

        # Console output
        print("\n" + "=" * 60)
        print("FIT VALIDATION METRICS")
        print("=" * 60)
        print(f"MAE (Mean Absolute Error):        {mae:.2f}")
        print(f"RMSE (Root Mean Squared Error):   {rmse:.2f}")
        print(f"MAPE (Mean Abs Percentage Error): {mape:.2f}%")
        print(f"R² (Coefficient of Determination): {r2:.4f}")
        print("=" * 60)

        return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

    def get_parameters(self) -> Dict[str, Any]:
        # Collect all estimated parameters into a dictionary
        out: Dict[str, Any] = {}
        if self.trend_params is not None:
            out["trend"] = self.trend_params.tolist()
        out["weekly"] = self.weekly_params
        out["yearly"] = self.yearly_params
        out["holiday"] = self.holiday_params
        out["spike"] = self.spike_params
        out["residual_cv"] = float(self.residual_cv)
        return out


class KolokwiumForecaster:
    """
    High-level forecaster built on PoissonProcessDecomposer.
    Handles loading data, diagnostics, forecasting, and saving results.
    """

    def __init__(self, data_path: str):
        # Path to input CSV file
        self.data_path = data_path

        # Containers for data, model, and forecast
        self.df: Optional[pd.DataFrame] = None
        self.decomposer: Optional[PoissonProcessDecomposer] = None
        self.forecast_df: Optional[pd.DataFrame] = None

    def load(self, sep: str = ";",
             date_col: str = "Date",
             value_col: str = "count") -> pd.DataFrame:
        # Load CSV data
        df = pd.read_csv(self.data_path, sep=sep, parse_dates=[date_col])

        # Sort and clean
        df = df.sort_values(date_col).reset_index(drop=True)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        # Standardize column name
        self.df = df.rename(columns={value_col: "count"})

        print(f"Loaded {len(self.df)} rows from {os.path.basename(self.data_path)}")
        return self.df

    def decompose(self, trend_method: str = "linear",
                  yearly_method: str = "fourier",
                  holiday_threshold: float = 0.5,
                  weekly_strength: float = 1.0,
                  spike_threshold: float = 1.4,
                  spike_scale: float = 0.5):
        # Ensure data is loaded
        if self.df is None:
            raise RuntimeError("Call load() first")

        # Prepare data for decomposer
        df2 = (
            self.df.reset_index()
            .rename(columns={self.df.index.name or "Date": "date",
                             "count": "calls"})[["date", "calls"]]
            .copy()
        )

        # Fit decomposition model
        model = PoissonProcessDecomposer(df2, date_col="date", value_col="calls")
        model.fit(trend_method=trend_method,
                  yearly_method=yearly_method,
                  holiday_threshold=holiday_threshold,
                  weekly_strength=weekly_strength,
                  spike_threshold=spike_threshold,
                  spike_scale=spike_scale)

        self.decomposer = model
        return model

    def diagnose_components(self) -> Dict[str, Any]:
        # Ensure model exists
        if self.decomposer is None:
            raise RuntimeError("Call decompose() first")

        model = self.decomposer
        lam = model.lambda_estimated if model.lambda_estimated is not None else np.ones(len(model.df))
        val = model.df[model.value_col].values

        # Basic summary statistics
        info = {
            "n": int(len(val)),
            "mean_obs": float(val.mean()),
            "mean_lambda": float(lam.mean()),
            "max_obs": float(val.max()),
            "max_lambda": float(lam.max()),
        }

        # Plot observed vs fitted intensity
        # plt.figure(figsize=(9, 3))
        # plt.plot(model.df["date"], val, alpha=0.6, label="observed")
        # plt.plot(model.df["date"], lam, alpha=0.9, label="fitted λ")
        # plt.legend(fontsize=8)
        # plt.title("Observed vs fitted λ")
        # plt.tight_layout()
        #plt.show()

        return info

    def estimate_monthly_seasonality_on(self, model: PoissonProcessDecomposer) -> Dict[int, float]:
        # Estimate monthly relative means from a given model
        dfm = model.df.copy()
        dfm["month"] = dfm["date"].dt.month
        vals = dfm.groupby("month")[model.value_col].mean().reindex(range(1, 13))
        overall = vals.mean()
        rel = (vals / (overall + 1e-12)).fillna(1.0)
        month_map = {int(m): float(rel.loc[m]) for m in rel.index}
        return month_map

    def estimate_monthly_seasonality(self):
        # Replace yearly component with monthly seasonality
        if self.decomposer is None:
            raise RuntimeError("Call decompose() first")

        month_map = self.estimate_monthly_seasonality_on(self.decomposer)
        self.decomposer.yearly_params = month_map
        self.decomposer.components["year_pattern"] = (
            self.decomposer.df["month"].map(month_map).values
        )

        # Plot monthly multipliers
        # plt.figure(figsize=(7, 2.5))
        # plt.bar(range(1, 13), [month_map[m] for m in range(1, 13)], color="C3")
        # plt.title("Monthly seasonality (relative)")
        # plt.tight_layout()
        # plt.show()

        return month_map

    def evaluate_seasonality_methods(self, holdout_days: int = 30) -> Dict[str, Any]:
        # Compare Fourier vs monthly seasonality on a holdout set
        if self.df is None:
            raise RuntimeError("Call load() first")

        full = self.df.reset_index()
        holdout_days = max(7, min(holdout_days, len(full) // 4))

        train = full.iloc[:-holdout_days].copy()
        hold = full.iloc[-holdout_days:].copy()

        df_train = train.rename(
            columns={train.columns[0]: "date", "count": "calls"}
        )[["date", "calls"]]

        m = PoissonProcessDecomposer(df_train, date_col="date", value_col="calls")
        m.fit(trend_method="linear", yearly_method="fourier", holiday_threshold=0.5)

        # Temporarily swap decomposer
        orig = self.decomposer
        self.decomposer = m

        # Fourier forecast
        rmse_fourier = np.inf
        if isinstance(m.yearly_params, dict) and "coefficients" in m.yearly_params:
            fA = self.forecast_from_decomposer(periods=holdout_days, calibrate=False)
            rmse_fourier = float(
                np.sqrt(np.mean((fA["expected_count"].values - hold["count"].values) ** 2))
            )

        # Monthly forecast
        month_map = self.estimate_monthly_seasonality_on(m)
        m.yearly_params = month_map
        m.components["year_pattern"] = m.df["month"].map(month_map).values
        fB = self.forecast_from_decomposer(periods=holdout_days, calibrate=False)
        rmse_monthly = float(
            np.sqrt(np.mean((fB["expected_count"].values - hold["count"].values) ** 2))
        )

        # Restore original model
        self.decomposer = orig

        best = "monthly" if rmse_monthly <= rmse_fourier else "fourier"
        return {
            "best": best,
            "rmse_fourier": rmse_fourier,
            "rmse_monthly": rmse_monthly
        }

    def calibrate_intensity(self, model: PoissonProcessDecomposer,
                            method: str = "mean") -> float:
        # Scale intensity to match observed data
        lam = model.lambda_estimated if model.lambda_estimated is not None else np.ones(len(model.df))
        obs = model.df[model.value_col].values

        if method == "mean":
            return float(obs.mean() / (lam.mean() + 1e-12))
        if method == "max":
            return float(obs.max() / (lam.max() + 1e-12))
        return 1.0

    def forecast_from_decomposer(self, periods: int = 365,
                                 random_seed: int = 0,
                                 calibrate: bool = True,
                                 calibrate_method: str = "mean",
                                 add_noise: bool = True,
                                 noise_scale: float = 0.15) -> pd.DataFrame:
        """
        Generate a future forecast using decomposed components.
        Optionally applies calibration and reduced multiplicative noise.
        """
        if self.decomposer is None:
            raise RuntimeError("Call decompose() first")

        model = self.decomposer

        # Future time indices and dates
        last_t = int(model.df["time_index"].max())
        future_idx = np.arange(last_t + 1, last_t + 1 + periods)
        future_dates = pd.date_range(
            model.df["date"].max() + pd.Timedelta(days=1),
            periods=periods,
            freq="D"
        )

        # Trend extrapolation
        if model.trend_params is not None:
            slope, intercept = model.trend_params[0], model.trend_params[1]
            trend_future = slope * future_idx + intercept
            trend_future = np.maximum(trend_future, 1e-6)
        else:
            last_trend = model.components.get(
                "trend",
                np.repeat(model.df[model.value_col].mean(), len(model.df))
            )
            trend_future = np.repeat(np.mean(last_trend[-14:]), periods)

        # Weekly pattern
        weekly = model.weekly_params or {}
        week_pattern_future = np.array(
            [weekly.get(int(d), 1.0) for d in future_dates.dayofweek]
        )

        # Yearly pattern
        yearp = model.yearly_params
        if isinstance(yearp, dict) and "coefficients" in yearp:
            coeffs = yearp["coefficients"]
            doy = future_dates.dayofyear.values
            Xf = np.column_stack([
                np.ones_like(doy),
                np.sin(2 * np.pi * doy / 365),
                np.cos(2 * np.pi * doy / 365)
            ])
            year_raw = Xf @ coeffs
            year_pattern_future = year_raw / (year_raw.mean() + 1e-12)
        elif isinstance(yearp, dict):
            year_pattern_future = np.array(
                [yearp.get(int(d.month), 1.0) for d in future_dates]
            )
        else:
            year_pattern_future = np.ones(periods)

        # Holiday effect
        hol = model.holiday_params or {}
        if "dates" in hol:
            md_set = {
                (pd.to_datetime(d).month, pd.to_datetime(d).day)
                for d in hol["dates"]
            }
            effect = hol.get("effect", 0.3)
            holiday_effect_future = np.array(
                [effect if (d.month, d.day) in md_set else 1.0 for d in future_dates]
            )
        else:
            holiday_effect_future = np.ones(periods)

        # Base intensity
        lam = (
            trend_future *
            week_pattern_future *
            year_pattern_future *
            holiday_effect_future
        )

        # Monthly spike adjustment
        spike_map = getattr(model, "spike_params", {}).get("month_map", {})
        if spike_map:
            month_mult = np.array(
                [spike_map.get(int(d.month), 1.0) for d in future_dates]
            )
            lam = lam * month_mult

        # Calibration
        if calibrate:
            scale = self.calibrate_intensity(model, method=calibrate_method)
            lam = lam * scale

        rng = np.random.default_rng(random_seed)

        # Optional multiplicative lognormal noise
        if add_noise:
            base_cv = getattr(model, "residual_cv", 0.1)
            used_cv = max(0.0, min(1.0, base_cv * float(noise_scale)))
            if used_cv > 1e-4:
                var_mult = used_cv ** 2
                sigma = np.sqrt(np.log(1 + var_mult))
                mu = -0.5 * sigma * sigma
                noise = rng.lognormal(mean=mu, sigma=sigma, size=periods)
                lam_noisy = lam * noise
            else:
                lam_noisy = lam
            lam_final = np.maximum(lam_noisy, 1e-6)
        else:
            lam_final = np.maximum(lam, 1e-6)

        # Build forecast DataFrame
        df_fore = (
            pd.DataFrame({
                "date": future_dates,
                "lambda": lam_final,
                "expected_count": lam_final
            })
            .set_index("date")
        )

        # Simulated Poisson counts
        df_fore["simulated_count"] = rng.poisson(lam_final)

        self.forecast_df = df_fore
        return df_fore

    def plot_history_and_forecast(self, show: bool = True, figsize=(11, 4)):
        # Plot historical data and forecast
        if self.df is None or self.forecast_df is None:
            raise RuntimeError("Need history and forecast")

        plt.figure(figsize=figsize)
        plt.plot(self.df.index, self.df["count"], label="history", alpha=0.7)
        plt.plot(self.forecast_df.index,
                 self.forecast_df["expected_count"],
                 label="expected",
                 color="C1")
        plt.plot(self.forecast_df.index,
                 self.forecast_df["simulated_count"],
                 label="simulated",
                 color="C2",
                 alpha=0.5)
        plt.axvline(self.df.index.max(), color="k", linestyle="--", alpha=0.5)
        plt.legend(fontsize=8)
        plt.tight_layout()
        if show:
            plt.show()

    def save_forecast(self,
                      out_name: str = "kolokwium_forecast_poisson_365.csv") -> str:
        # Save forecast to CSV
        if self.forecast_df is None:
            raise RuntimeError("No forecast to save")

        out_path = os.path.join(os.path.dirname(__file__), out_name)
        self.forecast_df.to_csv(out_path, index=True)
        print(f"Saved forecast to {out_path}")
        return out_path


if __name__ == "__main__":
    # Example execution script
    DATA_PATH = "/path/to/kolokwium_data.csv"

    # Initialize forecaster
    f = KolokwiumForecaster(DATA_PATH)

    # Load data
    f.load()

    # Decompose into components
    f.decompose(trend_method="linear",
                yearly_method="fourier",
                holiday_threshold=0.4)

    # Diagnostics and validation
    _ = f.diagnose_components()
    _ = f.decomposer.validate_fit()

    # Select best seasonality method
    res = f.evaluate_seasonality_methods(holdout_days=30)
    if res["best"] == "monthly":
        f.estimate_monthly_seasonality()

    # Generate forecast
    f.forecast_from_decomposer(periods=365,
                               calibrate=True,
                               calibrate_method="mean",
                               noise_scale=0.15)

    # Plot and save results
    f.plot_history_and_forecast()
    f.save_forecast()
