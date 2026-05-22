from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from inference_config import OUTPUT_FIGURES, OUTPUT_TABLES


def autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(values):
        return np.nan
    left = values[:-lag]
    right = values[lag:]
    if np.isclose(left.std(ddof=0), 0) or np.isclose(right.std(ddof=0), 0):
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def partial_autocorrelation_ols(values: np.ndarray, lag: int) -> float:
    if lag == 1:
        return autocorrelation(values, 1)
    if lag <= 0 or lag >= len(values) - 1:
        return np.nan
    y = values[lag:]
    x_cols = [values[lag - k : len(values) - k] for k in range(1, lag + 1)]
    x = np.column_stack([np.ones(len(y)), *x_cols])
    try:
        params = np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    return float(params[-1])


def adf_style_regression(values: np.ndarray) -> dict[str, float | str]:
    """Small ADF-style diagnostic without statsmodels p-values."""
    diff = np.diff(values)
    lagged_level = values[:-1]
    trend = np.arange(len(diff), dtype=float)
    x = np.column_stack([np.ones(len(diff)), trend, lagged_level])
    params, *_ = np.linalg.lstsq(x, diff, rcond=None)
    resid = diff - x @ params
    dof = max(len(diff) - x.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = float(np.sqrt(cov[2, 2]))
    t_stat = float(params[2] / se) if se > 0 else np.nan
    return {
        "metric": "adf_style_level_lag_t_stat",
        "value": t_stat,
        "interpretation": "more negative values indicate stronger mean reversion; p-value unavailable because statsmodels.tsa fails in this environment",
    }


def build_diagnostics(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    series = monthly["taxa_inadimplencia_pct"].to_numpy(dtype=float)
    dates = monthly["dt_referencia"]
    changes = np.diff(series)

    trend = stats.linregress(np.arange(len(series)), series)
    recent_3m = float(pd.Series(series).tail(3).mean())
    recent_6m = float(pd.Series(series).tail(6).mean())
    recent_12m = float(pd.Series(series).tail(12).mean())
    month_groups = [
        group["taxa_inadimplencia_pct"].to_numpy(dtype=float)
        for _, group in monthly.assign(month=dates.dt.month).groupby("month")
        if len(group) > 1
    ]
    seasonality_p = float(stats.f_oneway(*month_groups).pvalue) if len(month_groups) > 1 else np.nan

    summary = pd.DataFrame(
        [
            {"metric": "first_month", "value": dates.min().strftime("%Y-%m-%d"), "interpretation": "inicio da serie"},
            {"metric": "last_month", "value": dates.max().strftime("%Y-%m-%d"), "interpretation": "fim da serie observada"},
            {"metric": "months", "value": len(series), "interpretation": "amostra mensal curta; evitar modelos parametrizados demais"},
            {"metric": "mean_pct", "value": float(series.mean()), "interpretation": "nivel medio historico"},
            {"metric": "last_pct", "value": float(series[-1]), "interpretation": "ultima taxa observada"},
            {"metric": "last_3m_mean_pct", "value": recent_3m, "interpretation": "baseline primario Jan-Jun/2025"},
            {"metric": "last_6m_mean_pct", "value": recent_6m, "interpretation": "referencia recente suavizada"},
            {"metric": "last_12m_mean_pct", "value": recent_12m, "interpretation": "referencia anual suavizada"},
            {"metric": "monthly_change_mean_pp", "value": float(changes.mean()), "interpretation": "mudanca media mensal em p.p."},
            {"metric": "monthly_change_sd_pp", "value": float(changes.std(ddof=1)), "interpretation": "volatilidade mes a mes em p.p."},
            {"metric": "linear_trend_slope_pp_per_month", "value": float(trend.slope), "interpretation": "tendencia linear simples"},
            {"metric": "linear_trend_p_value", "value": float(trend.pvalue), "interpretation": "evidencia estatistica de tendencia linear"},
            {"metric": "calendar_month_anova_p_value", "value": seasonality_p, "interpretation": "triagem simples de sazonalidade por mes calendario"},
            adf_style_regression(series),
        ]
    )

    acf_rows = []
    for lag in range(1, min(13, len(series))):
        acf_rows.append(
            {
                "lag": lag,
                "acf": autocorrelation(series, lag),
                "pacf_ols": partial_autocorrelation_ols(series, lag),
            }
        )
    autocorr = pd.DataFrame(acf_rows)

    rolling = monthly[["dt_referencia", "taxa_inadimplencia_pct"]].copy()
    for window in [3, 6, 12]:
        rolling[f"rolling_mean_{window}m"] = rolling["taxa_inadimplencia_pct"].rolling(window).mean()
    rolling["monthly_change_pp"] = rolling["taxa_inadimplencia_pct"].diff()

    return summary, autocorr, rolling


def save_diagnostic_plots(monthly: pd.DataFrame, autocorr: pd.DataFrame, rolling: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(rolling["dt_referencia"], rolling["taxa_inadimplencia_pct"], color="#1f4e79", marker="o", label="Observado")
    ax.plot(rolling["dt_referencia"], rolling["rolling_mean_3m"], color="#8f2d56", linewidth=2, label="Media 3m")
    ax.plot(rolling["dt_referencia"], rolling["rolling_mean_6m"], color="#2a9d8f", linewidth=2, label="Media 6m")
    ax.plot(rolling["dt_referencia"], rolling["rolling_mean_12m"], color="#6c757d", linewidth=2, label="Media 12m")
    ax.set_title("Taxa mensal de inadimplencia e medias moveis")
    ax.set_xlabel("Mes de referencia")
    ax.set_ylabel("Taxa de inadimplencia (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "forecast_diagnostics_rolling_means.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(autocorr["lag"], autocorr["acf"], color="#1f4e79")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Autocorrelacao da inadimplencia mensal")
    ax.set_xlabel("Lag mensal")
    ax.set_ylabel("ACF")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "forecast_diagnostics_acf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_diagnostics(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, autocorr, rolling = build_diagnostics(monthly)
    summary.to_csv(OUTPUT_TABLES / "forecast_diagnostics_summary.csv", index=False)
    autocorr.to_csv(OUTPUT_TABLES / "forecast_diagnostics_autocorrelation.csv", index=False)
    rolling.to_csv(OUTPUT_TABLES / "forecast_diagnostics_rolling.csv", index=False)
    save_diagnostic_plots(monthly, autocorr, rolling)
    return summary, autocorr, rolling
