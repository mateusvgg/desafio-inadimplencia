from __future__ import annotations

import pandas as pd
import seaborn as sns

from forecasting_data import (
    load_monthly_delinquency_series,
    make_future_months,
)
from forecasting_diagnostics import save_diagnostics
from forecasting_models import (
    choose_model,
    create_final_forecast,
    evaluate_rolling_origin,
    get_forecast_specs,
    screen_exogenous_features,
)
from forecasting_plots import (
    save_backtest_plot,
    save_final_forecast_plot,
    save_model_comparison_plot,
)
from inference_config import OUTPUT_FIGURES, OUTPUT_TABLES


VALIDATION_STARTS = ["2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01"]
FORECAST_START = "2025-01-01"
FORECAST_HORIZON = 6


def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    monthly = load_monthly_delinquency_series()
    diagnostics_summary, diagnostics_autocorr, diagnostics_rolling = save_diagnostics(monthly)
    feature_screen = screen_exogenous_features(monthly)
    specs = get_forecast_specs(feature_screen)

    # Rolling-origin validation keeps the six-month executive horizon while
    # forcing every grid-search candidate to compete with the 3-month baseline.
    backtest_metrics, backtest_forecasts = evaluate_rolling_origin(
        monthly, VALIDATION_STARTS, specs, horizon=FORECAST_HORIZON
    )
    selected_model, selection_reason = choose_model(backtest_metrics)
    selected_errors = backtest_forecasts.loc[
        backtest_forecasts["model"].eq(selected_model), "error_pct"
    ]
    latest_fold = int(backtest_forecasts["fold"].max())
    latest_fold_forecasts = backtest_forecasts.loc[backtest_forecasts["fold"].eq(latest_fold)].copy()

    future_months = make_future_months(start=FORECAST_START, periods=FORECAST_HORIZON)
    final_forecast = create_final_forecast(
        monthly,
        specs,
        selected_model,
        future_months,
        validation_errors=selected_errors,
    )
    baseline_metrics = backtest_metrics.loc[
        backtest_metrics["model"].eq("baseline_mean_3m"), ["mae", "rmse", "bias"]
    ].iloc[0]
    decision_summary = pd.DataFrame(
        [
            {
                "selected_model": selected_model,
                "selection_rule": selection_reason,
                "validation_window": "rolling six-month origins from Apr/2024 to Jul/2024",
                "forecast_window": "2025-01-01 to 2025-06-01",
                "primary_baseline": "mean delinquency rate from the last 3 observed months",
                "baseline_mae": baseline_metrics["mae"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_bias": baseline_metrics["bias"],
                "feature_policy": (
                    "macro and portfolio features screened by lagged correlation; "
                    "future exogenous values held at last observed level when used"
                ),
                "business_interpretation": (
                    "forecast starts from the recent portfolio level and only moves "
                    "away from the 3-month mean when validation supports the added structure"
                ),
            }
        ]
    )

    monthly.to_csv(OUTPUT_TABLES / "forecast_monthly_series.csv", index=False)
    feature_screen.to_csv(OUTPUT_TABLES / "forecast_exogenous_feature_screen.csv", index=False)
    backtest_metrics.to_csv(OUTPUT_TABLES / "forecast_backtest_metrics.csv", index=False)
    backtest_forecasts.to_csv(OUTPUT_TABLES / "forecast_backtest_predictions.csv", index=False)
    latest_fold_forecasts.to_csv(
        OUTPUT_TABLES / "forecast_latest_fold_predictions.csv", index=False
    )
    final_forecast.to_csv(OUTPUT_TABLES / "forecast_jan_jun_2025.csv", index=False)
    decision_summary.to_csv(OUTPUT_TABLES / "forecast_model_decision_summary.csv", index=False)

    save_backtest_plot(latest_fold_forecasts, backtest_metrics)
    save_model_comparison_plot(backtest_metrics)
    save_final_forecast_plot(monthly, final_forecast)

    print("Diagnostics summary:")
    print(diagnostics_summary.to_string(index=False))
    print("Autocorrelation diagnostic:")
    print(diagnostics_autocorr.head(12).to_string(index=False))
    print("Rolling diagnostic tail:")
    print(diagnostics_rolling.tail(6).to_string(index=False))
    print("Top exogenous feature screen:")
    print(
        feature_screen.sort_values("abs_best_corr", ascending=False)
        .head(12)
        .to_string(index=False)
    )
    print("Selected forecast model:", selected_model)
    print("Backtest metrics:")
    print(backtest_metrics.to_string(index=False))
    print("Forecast Jan-Jun/2025:")
    print(final_forecast.to_string(index=False))


if __name__ == "__main__":
    main()
