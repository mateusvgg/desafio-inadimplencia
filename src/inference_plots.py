from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from inference_config import OUTPUT_FIGURES
from inference_data import add_contract_features


def save_contract_effect_plot(effects: pd.DataFrame) -> None:
    plot = effects.copy()
    plot["abs_effect"] = plot["marginal_effect_pp"].abs()
    plot = plot.nlargest(10, "abs_effect").sort_values("marginal_effect_pp")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#8f2d56" if v > 0 else "#287271" for v in plot["marginal_effect_pp"]]
    ax.barh(plot["factor"], plot["marginal_effect_pp"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Efeito marginal medio na probabilidade de inadimplencia")
    ax.set_xlabel("Diferenca em pontos percentuais")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(
        min(0, plot["marginal_effect_pp"].min()) - 0.7,
        max(0, plot["marginal_effect_pp"].max()) + 0.7,
    )
    for y, value in enumerate(plot["marginal_effect_pp"]):
        ha = "left" if value >= 0 else "right"
        offset = 0.04 if value >= 0 else -0.04
        ax.text(value + offset, y, f"{value:+.2f} p.p.", va="center", ha=ha, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "inference_marginal_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_odds_ratio_plot(results_table: pd.DataFrame) -> None:
    selected_terms = {
        "score_100": "Score +100 pontos",
        "ltv_10pp": "LTV +10 p.p.",
        "tx_juros_anual": "Taxa juros +1 p.p.",
        "prazo_5anos": "Prazo +5 anos",
        "log_vr_financiado": "Valor financiado",
        'C(faixa_renda, Treatment(reference="Acima de 10 SM"))[T.Até 3 SM]': "Renda ate 3 SM",
        'C(faixa_renda, Treatment(reference="Acima de 10 SM"))[T.3 a 6 SM]': "Renda 3 a 6 SM",
        'C(programa_social, Treatment(reference="Livre"))[T.FGTS]': "FGTS",
        'C(programa_social, Treatment(reference="Livre"))[T.MCMV]': "MCMV",
    }
    plot = results_table.loc[results_table["term"].isin(selected_terms)].copy()
    plot["label"] = plot["term"].map(selected_terms)
    plot = plot.sort_values("odds_ratio")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(
        plot["odds_ratio"],
        plot["label"],
        xerr=[
            plot["odds_ratio"] - plot["or_ci_low"],
            plot["or_ci_high"] - plot["odds_ratio"],
        ],
        fmt="o",
        color="#1f4e79",
        ecolor="#8aa6c1",
        capsize=3,
    )
    ax.axvline(1, color="black", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_title("Razoes de chance ajustadas com IC 95%")
    ax.set_xlabel("Odds ratio, escala log")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "inference_odds_ratios.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_risk_scenario_plot(result, contract_base: pd.DataFrame) -> pd.DataFrame:
    scenarios = pd.DataFrame(
        [
            {
                "scenario": "Baixo risco",
                "score_credito_contratacao": 800,
                "ltv": 70,
                "tx_juros_anual": 7.0,
                "prazo_meses": 240,
                "vr_financiado": 180000,
                "faixa_renda": "Acima de 10 SM",
                "tipo_imovel": "Novo",
                "programa_social": "Livre",
                "uf": "SP",
            },
            {
                "scenario": "Perfil medio",
                "score_credito_contratacao": contract_base["score_credito_contratacao"].median(),
                "ltv": contract_base["ltv"].median(),
                "tx_juros_anual": contract_base["tx_juros_anual"].median(),
                "prazo_meses": contract_base["prazo_meses"].median(),
                "vr_financiado": contract_base["vr_financiado"].median(),
                "faixa_renda": "3 a 6 SM",
                "tipo_imovel": "Usado",
                "programa_social": "FGTS",
                "uf": "SP",
            },
            {
                "scenario": "Alto risco",
                "score_credito_contratacao": 520,
                "ltv": 89,
                "tx_juros_anual": 10.0,
                "prazo_meses": 360,
                "vr_financiado": 260000,
                "faixa_renda": "Até 3 SM",
                "tipo_imovel": "Usado",
                "programa_social": "MCMV",
                "uf": "SP",
            },
        ]
    )
    scenarios["id_contrato"] = range(1, len(scenarios) + 1)
    scenarios["dt_contratacao"] = pd.Timestamp("2024-01-01")
    scenarios["vr_entrada"] = scenarios["vr_financiado"] * (100 - scenarios["ltv"]) / scenarios["ltv"]
    scenarios = add_contract_features(scenarios)
    scenarios["probabilidade_inadimplencia"] = result.predict(scenarios)
    scenarios["probabilidade_pct"] = scenarios["probabilidade_inadimplencia"] * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(scenarios["scenario"], scenarios["probabilidade_pct"], color=["#287271", "#f4a261", "#8f2d56"])
    ax.set_title("Probabilidade ajustada por perfil de contrato")
    ax.set_ylabel("Probabilidade ever-inadimplente (%)")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(scenarios["probabilidade_pct"]):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "inference_risk_scenarios.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return scenarios


def save_calibration_plot(calibration: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(calibration["decile"], calibration["observed_rate"] * 100, marker="o", label="Observado")
    ax.plot(calibration["decile"], calibration["predicted_rate"] * 100, marker="o", label="Previsto")
    ax.set_title("Calibracao por decis de risco - modelo contratual")
    ax.set_xlabel("Decil de risco previsto")
    ax.set_ylabel("Taxa de inadimplencia (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "inference_contract_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_hazard_effect_plot(effects: pd.DataFrame) -> None:
    plot = effects.sort_values("marginal_effect_pp")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#8f2d56" if v > 0 else "#287271" for v in plot["marginal_effect_pp"]]
    ax.barh(plot["factor"], plot["marginal_effect_pp"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Efeito medio na entrada mensal em inadimplencia")
    ax.set_xlabel("Diferenca em pontos percentuais")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    for y, value in enumerate(plot["marginal_effect_pp"]):
        ha = "left" if value >= 0 else "right"
        offset = 0.005 if value >= 0 else -0.005
        ax.text(value + offset, y, f"{value:+.3f} p.p.", va="center", ha=ha, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "inference_hazard_macro_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
