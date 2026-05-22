from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from inference_config import OUTPUT_FIGURES


def save_backtest_plot(backtest_forecasts: pd.DataFrame, metrics: pd.DataFrame, top_n: int = 4) -> None:
    top_models = metrics.head(top_n)["model"].tolist()
    plot_forecasts = backtest_forecasts.loc[backtest_forecasts["model"].isin(top_models)].copy()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    actual = plot_forecasts.drop_duplicates("dt_referencia")
    ax.plot(
        actual["dt_referencia"],
        actual["actual_pct"],
        color="black",
        marker="o",
        linewidth=2.5,
        label="Observado",
    )

    for model, subset in plot_forecasts.groupby("model"):
        ax.plot(
            subset["dt_referencia"],
            subset["forecast_pct"],
            linewidth=1.6,
            alpha=0.75,
            label=model,
        )

    ax.set_title(f"Backtest dos {top_n} melhores modelos - Jul a Dez/2024")
    ax.set_xlabel("Mes de referencia")
    ax.set_ylabel("Taxa de inadimplencia (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "forecast_backtest_models.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_plot(metrics: pd.DataFrame) -> None:
    plot = metrics.sort_values("mae", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(plot["model"], plot["mae"], color="#1f4e79")
    ax.invert_yaxis()
    ax.set_title("Comparacao dos modelos por erro absoluto medio")
    ax.set_xlabel("MAE em pontos percentuais")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    for y, value in enumerate(plot["mae"]):
        ax.text(value + 0.01, y, f"{value:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "forecast_model_comparison_mae.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_final_forecast_plot(monthly: pd.DataFrame, forecast: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        monthly["dt_referencia"],
        monthly["taxa_inadimplencia_pct"],
        color="#1f4e79",
        marker="o",
        linewidth=2,
        label="Historico observado",
    )
    ax.plot(
        forecast["dt_referencia"],
        forecast["forecast_pct"],
        color="#8f2d56",
        marker="o",
        linewidth=2.5,
        label="Previsao Jan-Jun/2025",
    )
    ax.fill_between(
        forecast["dt_referencia"],
        forecast["lower_95_pct"],
        forecast["upper_95_pct"],
        color="#8f2d56",
        alpha=0.18,
        label="Intervalo 95%",
    )
    ax.axvline(pd.Timestamp("2025-01-01"), color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_title("Previsao da taxa mensal de inadimplencia")
    ax.set_xlabel("Mes de referencia")
    ax.set_ylabel("Taxa de inadimplencia (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "forecast_final_jan_jun_2025.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
