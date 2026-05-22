from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


TARGET_COL = "taxa_inadimplencia_pct"
EXOGENOUS_CANDIDATES = [
    "selic_12m",
    "ipca_12m",
    "tx_desemprego_12m",
    "rendimento_medio",
    "confianca_consumidor",
    "contratos_ativos",
    "dias_atraso_medio",
    "saldo_devedor_medio",
    "pct_pago_devido",
    "share_atraso_1_90",
    "ltv_medio",
    "score_medio",
    "share_baixa_renda",
    "share_programa_social",
]
MACRO_FEATURES = {
    "selic_12m",
    "ipca_12m",
    "tx_desemprego_12m",
    "rendimento_medio",
    "confianca_consumidor",
}


@dataclass(frozen=True)
class ForecastSpec:
    model: str
    family: str
    description: str
    params: dict[str, float | int | str | None]
    forecast_function: Callable[[pd.DataFrame, int], np.ndarray]


def clip_rate(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), 0.0, 100.0)


def series_from_frame(train: pd.DataFrame) -> pd.Series:
    return train[TARGET_COL].astype(float)


def naive_last(train: pd.DataFrame, horizon: int) -> np.ndarray:
    series = series_from_frame(train)
    return np.repeat(float(series.iloc[-1]), horizon)


def recent_mean(train: pd.DataFrame, horizon: int, window: int) -> np.ndarray:
    series = series_from_frame(train)
    return np.repeat(float(series.tail(window).mean()), horizon)


def drift(train: pd.DataFrame, horizon: int, window: int | None) -> np.ndarray:
    series = series_from_frame(train)
    if window is not None:
        series = series.tail(window)
    if len(series) < 2:
        return naive_last(train, horizon)
    monthly_drift = (float(series.iloc[-1]) - float(series.iloc[0])) / (len(series) - 1)
    return clip_rate(float(series.iloc[-1]) + monthly_drift * np.arange(1, horizon + 1))


def linear_trend(train: pd.DataFrame, horizon: int, window: int | None) -> np.ndarray:
    series = series_from_frame(train)
    if window is not None:
        series = series.tail(window)
    if len(series) < 3:
        return naive_last(train, horizon)
    t = np.arange(len(series), dtype=float)
    slope, intercept = np.polyfit(t, series.to_numpy(dtype=float), deg=1)
    future_t = np.arange(len(series), len(series) + horizon, dtype=float)
    return clip_rate(intercept + slope * future_t)


def ses(train: pd.DataFrame, horizon: int, alpha: float) -> np.ndarray:
    values = series_from_frame(train).to_numpy(dtype=float)
    level = values[0]
    for value in values:
        level = alpha * value + (1 - alpha) * level
    return clip_rate(np.repeat(level, horizon))


def holt(train: pd.DataFrame, horizon: int, alpha: float, beta: float, phi: float) -> np.ndarray:
    values = series_from_frame(train).to_numpy(dtype=float)
    if len(values) < 3:
        return naive_last(train, horizon)
    level = values[0]
    trend = values[1] - values[0]
    for value in values[1:]:
        forecast = level + phi * trend
        previous_level = level
        level = alpha * value + (1 - alpha) * forecast
        trend = beta * (level - previous_level) + (1 - beta) * phi * trend
    steps = np.arange(1, horizon + 1, dtype=float)
    damped_steps = np.array([sum(phi**i for i in range(1, int(step) + 1)) for step in steps])
    return clip_rate(level + damped_steps * trend)


def ar_diff(train: pd.DataFrame, horizon: int, p: int) -> np.ndarray:
    values = series_from_frame(train).to_numpy(dtype=float)
    diffs = np.diff(values)
    if p == 0:
        next_diff = float(diffs.mean()) if len(diffs) else 0.0
        return clip_rate(values[-1] + next_diff * np.arange(1, horizon + 1))
    if len(diffs) <= p + 2:
        return naive_last(train, horizon)

    y = diffs[p:]
    x_cols = [diffs[p - lag : len(diffs) - lag] for lag in range(1, p + 1)]
    x = np.column_stack([np.ones(len(y)), *x_cols])
    try:
        params = np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return naive_last(train, horizon)

    history = list(diffs[-p:])
    level = float(values[-1])
    forecasts = []
    for _ in range(horizon):
        lag_values = np.array(history[-p:][::-1])
        next_diff = float(params[0] + params[1:] @ lag_values)
        level += next_diff
        forecasts.append(level)
        history.append(next_diff)
    return clip_rate(np.array(forecasts))


def _standardize_train_future(
    train: pd.DataFrame, feature_cols: list[str], horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = train[feature_cols].astype(float)
    means = train_features.mean()
    stds = train_features.std(ddof=0).replace(0, 1)
    x_train = (train_features - means) / stds
    latest = train_features.iloc[-1]
    future = pd.DataFrame([latest.to_dict()] * horizon)
    x_future = (future - means) / stds
    return x_train, x_future


def dynamic_regression(
    train: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
    y_lags: int,
) -> np.ndarray:
    available = [col for col in feature_cols if col in train.columns and train[col].notna().all()]
    if len(train) <= y_lags + len(available) + 3:
        return recent_mean(train, horizon, 3)

    y = series_from_frame(train).reset_index(drop=True)
    x_train, x_future = _standardize_train_future(train, available, horizon)
    rows = []
    target = []
    for idx in range(y_lags, len(train)):
        lag_values = {f"lag_{lag}": y.iloc[idx - lag] for lag in range(1, y_lags + 1)}
        feature_values = x_train.iloc[idx].to_dict()
        rows.append({**lag_values, **feature_values})
        target.append(y.iloc[idx])

    design = pd.DataFrame(rows)
    x = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])
    try:
        params = np.linalg.lstsq(x, np.array(target, dtype=float), rcond=None)[0]
    except np.linalg.LinAlgError:
        return recent_mean(train, horizon, 3)

    history = y.tolist()
    forecasts = []
    for step in range(horizon):
        lag_values = [history[-lag] for lag in range(1, y_lags + 1)]
        future_values = x_future.iloc[step].to_numpy(dtype=float).tolist()
        row = np.array([1.0, *lag_values, *future_values])
        prediction = float(row @ params)
        prediction = float(clip_rate(np.array([prediction]))[0])
        forecasts.append(prediction)
        history.append(prediction)
    return np.array(forecasts)


def make_spec(
    model: str,
    family: str,
    description: str,
    params: dict[str, float | int | str | None],
    forecast_function: Callable[[pd.DataFrame, int], np.ndarray],
) -> ForecastSpec:
    return ForecastSpec(model, family, description, params, forecast_function)


def get_forecast_specs(feature_screen: pd.DataFrame | None = None) -> list[ForecastSpec]:
    specs: list[ForecastSpec] = [
        make_spec("baseline_mean_3m", "baseline", "Media dos ultimos 3 meses", {"window": 3}, lambda train, h: recent_mean(train, h, 3)),
        make_spec("benchmark_last", "benchmark", "Ultima taxa observada constante", {}, naive_last),
    ]

    for window in [3, 6, 9, 12]:
        specs.append(
            make_spec(
                f"ma_{window}m",
                "moving_average",
                f"Media movel de {window} meses",
                {"window": window},
                lambda train, h, w=window: recent_mean(train, h, w),
            )
        )
    for window in [6, 12, 18, 24, None]:
        label = "full" if window is None else f"{window}m"
        specs.append(
            make_spec(
                f"drift_{label}",
                "drift",
                f"Drift historico ({label})",
                {"window": window},
                lambda train, h, w=window: drift(train, h, w),
            )
        )
        specs.append(
            make_spec(
                f"trend_{label}",
                "linear_trend",
                f"Tendencia linear ({label})",
                {"window": window},
                lambda train, h, w=window: linear_trend(train, h, w),
            )
        )
    for alpha in [0.2, 0.4, 0.6, 0.8]:
        specs.append(
            make_spec(
                f"ses_a{int(alpha * 10)}",
                "ses_grid",
                f"Suavizacao exponencial alpha={alpha:.1f}",
                {"alpha": alpha},
                lambda train, h, a=alpha: ses(train, h, a),
            )
        )
    for alpha in [0.3, 0.6]:
        for beta in [0.1, 0.3]:
            for phi in [0.85, 0.95, 1.0]:
                specs.append(
                    make_spec(
                        f"holt_a{int(alpha*10)}_b{int(beta*10)}_p{int(phi*100)}",
                        "holt_grid",
                        f"Holt alpha={alpha:.1f}, beta={beta:.1f}, phi={phi:.2f}",
                        {"alpha": alpha, "beta": beta, "phi": phi},
                        lambda train, h, a=alpha, b=beta, p=phi: holt(train, h, a, b, p),
                    )
                )
    for p in [0, 1, 2]:
        specs.append(
            make_spec(
                f"ar_diff_p{p}",
                "arima_manual_grid",
                f"AR manual em diferencas p={p}",
                {"p": p, "d": 1},
                lambda train, h, order=p: ar_diff(train, h, order),
            )
        )

    if feature_screen is not None and not feature_screen.empty:
        feature_rank = (
            feature_screen[["feature", "abs_best_corr"]]
            .drop_duplicates()
            .sort_values("abs_best_corr", ascending=False)
        )
        macro_features = feature_rank.loc[feature_rank["feature"].isin(MACRO_FEATURES), "feature"].head(2).tolist()
        portfolio_features = feature_rank.loc[~feature_rank["feature"].isin(MACRO_FEATURES), "feature"].head(2).tolist()
        feature_sets = []
        if macro_features:
            feature_sets.append(("macro", macro_features))
        if portfolio_features:
            feature_sets.append(("portfolio", portfolio_features))
        if macro_features and portfolio_features:
            feature_sets.append(("macro_portfolio", macro_features + portfolio_features))

        for label, features in feature_sets:
            for y_lags in [1, 2]:
                specs.append(
                    make_spec(
                        f"dynreg_{label}_ylag{y_lags}",
                        "dynamic_regression_grid",
                        f"Regressao dinamica com {label}; exogenas constantes no futuro",
                        {"features": ",".join(features), "y_lags": y_lags, "future_exog": "last_observed_constant"},
                        lambda train, h, cols=features, lags=y_lags: dynamic_regression(train, h, cols, lags),
                    )
                )

    return specs


def screen_exogenous_features(
    monthly: pd.DataFrame,
    features: list[str] | None = None,
    max_lag: int = 3,
) -> pd.DataFrame:
    features = features or EXOGENOUS_CANDIDATES
    rows = []
    y = monthly[TARGET_COL].astype(float)
    for feature in features:
        if feature not in monthly.columns:
            continue
        x = monthly[feature].astype(float)
        lag_corrs = []
        for lag in range(0, max_lag + 1):
            shifted = x.shift(lag)
            valid = y.notna() & shifted.notna()
            corr = float(y[valid].corr(shifted[valid])) if valid.sum() >= 8 else np.nan
            rows.append(
                {
                    "feature": feature,
                    "lag_months": lag,
                    "correlation": corr,
                    "abs_correlation": abs(corr) if pd.notna(corr) else np.nan,
                    "future_assumption": "last observed value held constant for forecasting",
                }
            )
            lag_corrs.append(abs(corr) if pd.notna(corr) else np.nan)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    best = (
        result.groupby("feature", as_index=False)["abs_correlation"]
        .max()
        .rename(columns={"abs_correlation": "abs_best_corr"})
    )
    return result.merge(best, on="feature", how="left")


def evaluate_specs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    specs: list[ForecastSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon = len(validation)
    actual = validation[TARGET_COL].to_numpy(dtype=float)
    forecast_rows = []
    metric_rows = []

    for spec in specs:
        predictions = spec.forecast_function(train, horizon)
        errors = actual - predictions
        forecast_rows.extend(
            {
                "model": spec.model,
                "family": spec.family,
                "description": spec.description,
                "params": str(spec.params),
                "dt_referencia": date,
                "actual_pct": observed,
                "forecast_pct": predicted,
                "error_pct": error,
            }
            for date, observed, predicted, error in zip(
                validation["dt_referencia"], actual, predictions, errors
            )
        )
        metric_rows.append(
            {
                "model": spec.model,
                "family": spec.family,
                "description": spec.description,
                "params": str(spec.params),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "bias": float(np.mean(predictions - actual)),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values(["mae", "rmse"])
    forecasts = pd.DataFrame(forecast_rows)
    return metrics, forecasts


def evaluate_rolling_origin(
    monthly: pd.DataFrame,
    validation_starts: list[str],
    specs: list[ForecastSpec],
    horizon: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_forecasts = []

    for fold_number, validation_start in enumerate(validation_starts, start=1):
        start = pd.Timestamp(validation_start)
        validation_months = pd.date_range(start=start, periods=horizon, freq="MS")
        train = monthly.loc[monthly["dt_referencia"] < start].copy()
        validation = monthly.loc[monthly["dt_referencia"].isin(validation_months)].copy()
        if len(validation) != horizon:
            continue

        _, fold_forecasts = evaluate_specs(train, validation, specs)
        fold_forecasts["fold"] = fold_number
        fold_forecasts["validation_start"] = start
        all_forecasts.append(fold_forecasts)

    forecasts = pd.concat(all_forecasts, ignore_index=True)
    metrics = (
        forecasts.groupby(["model", "family", "description", "params"], as_index=False)
        .agg(
            mae=("error_pct", lambda s: float(np.mean(np.abs(s)))),
            rmse=("error_pct", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            bias=("error_pct", lambda s: float(np.mean(-s))),
            folds=("fold", "nunique"),
            observations=("error_pct", "size"),
        )
        .sort_values(["mae", "rmse"])
    )
    return metrics, forecasts


def choose_model(metrics: pd.DataFrame, minimum_improvement_pp: float = 0.03) -> tuple[str, str]:
    ordered = metrics.sort_values(["mae", "rmse"]).reset_index(drop=True)
    baseline = ordered.loc[ordered["model"].eq("baseline_mean_3m")].iloc[0]
    best = ordered.iloc[0]
    improvement = float(baseline["mae"] - best["mae"])

    if str(best["model"]) == "baseline_mean_3m":
        return "baseline_mean_3m", "3-month mean retained because no model improved validation MAE"
    if improvement >= minimum_improvement_pp:
        return (
            str(best["model"]),
            f"lowest rolling-origin MAE and improves over baseline_mean_3m by {improvement:.3f} p.p.",
        )
    return (
        "baseline_mean_3m",
        f"best model improved MAE by only {improvement:.3f} p.p.; retained simpler 3-month mean baseline",
    )


def create_final_forecast(
    monthly: pd.DataFrame,
    specs: list[ForecastSpec],
    selected_model: str,
    future_months: pd.DatetimeIndex,
    validation_errors: pd.Series,
) -> pd.DataFrame:
    spec = next(s for s in specs if s.model == selected_model)
    forecast = spec.forecast_function(monthly, len(future_months))
    residual_sigma = float(validation_errors.std(ddof=1))
    interval_width = 1.96 * residual_sigma if np.isfinite(residual_sigma) else 0.0

    return pd.DataFrame(
        {
            "dt_referencia": future_months,
            "selected_model": selected_model,
            "model_family": spec.family,
            "forecast_pct": forecast,
            "lower_95_pct": clip_rate(forecast - interval_width),
            "upper_95_pct": clip_rate(forecast + interval_width),
            "interval_method": "rolling-origin validation residual normal approximation",
        }
    )
