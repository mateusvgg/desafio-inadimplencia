# Forecast Methodology Reset

## Why step back

The existing forecast pipeline is directionally sound because it uses classical, interpretable methods and rolling-origin validation. However, it moves too quickly from the monthly delinquency series to a broad model comparison. For the executive question, the forecast should be easier to defend if the statistical properties of the series are documented before model choice.

The main risk in the current approach is not that the models are too complex. The risk is that the selection story is cluttered: several candidates are compared, but the report does not first explain whether delinquency behaves like a stable level, a trend, a seasonal series, a shock-reverting process, or a macro-sensitive series.

## Replacement principle

Use a narrower, staged workflow:

1. Diagnose the delinquency series.
2. Establish a naive baseline using the mean of the last 3 months.
3. Test whether simple time-series structure improves on that baseline.
4. Test whether macro and portfolio features improve the forecast enough to justify inclusion.
5. Select the final model using rolling-origin validation and business interpretability.

## Statistical properties to document first

- Monthly active-contract denominator and delinquency-rate definition.
- Level, trend, month-to-month changes, volatility, and outlier months.
- Rolling 3-month, 6-month, and 12-month means to identify recent direction.
- Autocorrelation and partial autocorrelation, interpreted cautiously because the sample is short.
- Stationarity diagnostics using ADF and KPSS where feasible.
- Simple seasonality screen by calendar month.
- Structural-change or late-period shift checks, especially around 2024.
- Correlations and lagged correlations with macro indicators.
- Correlations with aggregate portfolio mix features, if available from contract-month data.

## Baseline

The required first benchmark should be:

`forecast Jan-Jun/2025 = mean delinquency rate from Oct-Dec/2024`.

This baseline is intentionally simple. It answers: if CAIXA assumes the recent portfolio condition persists, what happens? Every later model must beat this baseline on validation or add a clearly useful business interpretation.

## Candidate models

Candidate models should remain classical and interpretable:

- Last-observation naive forecast, as a secondary benchmark.
- Last-3-month mean baseline, as the primary benchmark.
- Moving averages with grid-searched windows.
- Linear trend models with grid-searched recent windows.
- Exponential smoothing and Holt variants with parameter grid search or transparent optimization.
- ARIMA or SARIMAX with small grid search over low-order specifications.
- Dynamic regression or SARIMAX with exogenous variables, only after lag screening.

## Exogenous features to consider

Features besides delinquency should be considered in two groups:

- Macro indicators: Selic, IPCA, unemployment, average real income, and consumer confidence, using documented lags.
- Portfolio composition: active contracts, average outstanding balance, average days past due, share of early arrears if available, average paid-to-due ratio, and mix variables by LTV, score, income band, social program, or region if they can be aggregated without leakage.

For the January-June 2025 forecast, future exogenous values must be either known, scenario-based, held constant, or separately forecasted. The assumption must be documented.

## Model-selection rule

Use rolling-origin validation with six-month horizons when possible. Compare MAE and RMSE in percentage points, plus bias. The selected model should satisfy three conditions:

- It improves meaningfully over the 3-month-mean baseline, or it is retained as the best transparent baseline.
- It has a plausible business story.
- It does not rely on future information unavailable at forecast time.

## Documentation rule

Each stage should save tables and figures before moving on:

- Diagnostics tables and plots.
- Baseline forecast and validation errors.
- Grid-search results.
- Exogenous-feature screening.
- Final model decision summary.
- Final January-June 2025 forecast with uncertainty intervals.

