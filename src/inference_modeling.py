from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial

from inference_config import MACRO_COLS


CONTRACT_FORMULA = (
    "fl_ever_inadimplente ~ "
    "score_100 + ltv_10pp + tx_juros_anual + prazo_5anos + log_vr_financiado + "
    'C(faixa_renda, Treatment(reference="Acima de 10 SM")) + '
    'C(tipo_imovel, Treatment(reference="Novo")) + '
    'C(programa_social, Treatment(reference="Livre")) + '
    'C(uf, Treatment(reference="SP"))'
)

HAZARD_BASE_FORMULA = (
    "entrada_inadimplencia ~ "
    "score_100 + ltv_10pp + tx_juros_anual + prazo_5anos + log_vr_financiado + "
    "meses_desde_contratacao + I(meses_desde_contratacao ** 2) + "
    'C(faixa_renda, Treatment(reference="Acima de 10 SM")) + '
    'C(tipo_imovel, Treatment(reference="Novo")) + '
    'C(programa_social, Treatment(reference="Livre")) + '
    'C(regiao, Treatment(reference="Sudeste"))'
)


def auc_score(y_true: pd.Series, y_score: pd.Series) -> float:
    data = pd.DataFrame({"y": y_true.to_numpy(), "score": y_score.to_numpy()}).sort_values("score")
    n_pos = data["y"].sum()
    n_neg = len(data) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.arange(1, len(data) + 1), index=data.index)
    rank_sum_pos = ranks.loc[data["y"].eq(1)].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def tidy_glm_result(result, model_name: str) -> pd.DataFrame:
    conf = result.conf_int()
    return pd.DataFrame(
        {
            "model": model_name,
            "term": result.params.index,
            "coef": result.params.values,
            "std_error": result.bse.values,
            "p_value": result.pvalues.values,
            "odds_ratio": np.exp(result.params.values),
            "or_ci_low": np.exp(conf[0].values),
            "or_ci_high": np.exp(conf[1].values),
        }
    )


def predict_with_changes(result, base: pd.DataFrame, changes: dict[str, object]) -> pd.Series:
    scenario = base.copy()
    for col, value in changes.items():
        scenario[col] = value(scenario[col]) if callable(value) else value
    return result.predict(scenario)


def average_probability_delta(
    result,
    base: pd.DataFrame,
    changes: dict[str, object],
    label: str,
    group: str,
) -> dict[str, float | str]:
    p0 = result.predict(base)
    p1 = predict_with_changes(result, base, changes)
    return {
        "group": group,
        "factor": label,
        "marginal_effect_pp": (p1.mean() - p0.mean()) * 100,
        "base_probability_pct": p0.mean() * 100,
        "scenario_probability_pct": p1.mean() * 100,
    }


def average_category_contrast(
    result,
    base: pd.DataFrame,
    column: str,
    reference_value: str,
    scenario_value: str,
    label: str,
    group: str,
) -> dict[str, float | str]:
    reference = base.copy()
    scenario = base.copy()
    reference[column] = reference_value
    scenario[column] = scenario_value
    p0 = result.predict(reference)
    p1 = result.predict(scenario)
    return {
        "group": group,
        "factor": label,
        "marginal_effect_pp": (p1.mean() - p0.mean()) * 100,
        "base_probability_pct": p0.mean() * 100,
        "scenario_probability_pct": p1.mean() * 100,
    }


def run_contract_model(contract_base: pd.DataFrame):
    # HC3 is a conservative robust covariance choice for the cross-sectional
    # contract model; it keeps inference less sensitive to heteroskedasticity.
    result = smf.glm(CONTRACT_FORMULA, contract_base, family=Binomial()).fit(cov_type="HC3")
    pred = result.predict(contract_base)
    diagnostics = {
        "model": "contract_logit_ever_delinquent",
        "rows": len(contract_base),
        "events": int(contract_base["fl_ever_inadimplente"].sum()),
        "event_rate": float(contract_base["fl_ever_inadimplente"].mean()),
        "aic": float(result.aic),
        "auc": auc_score(contract_base["fl_ever_inadimplente"], pred),
    }
    return result, pred, diagnostics


def run_hazard_model(
    hazard_base: pd.DataFrame, macro_term: str, excluded_left_censored_contracts: int
):
    model_data = hazard_base.dropna(subset=[macro_term]).copy()
    groups = model_data["dt_referencia"].dt.to_period("M").astype(str)
    result = smf.glm(
        f"{HAZARD_BASE_FORMULA} + {macro_term}",
        model_data,
        family=Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": groups})
    pred = result.predict(model_data)
    diagnostics = {
        "model": f"hazard_logit_entry_{macro_term}",
        "rows": len(model_data),
        "events": int(model_data["entrada_inadimplencia"].sum()),
        "event_rate": float(model_data["entrada_inadimplencia"].mean()),
        "months": int(model_data["dt_referencia"].nunique()),
        "excluded_left_censored_contracts": int(excluded_left_censored_contracts),
        "aic": float(result.aic),
        "auc": auc_score(model_data["entrada_inadimplencia"], pred),
    }
    return result, model_data, pred, diagnostics


def screen_macro_terms(hazard_base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidate_terms = [
        *(f"{col}_lag{lag}" for col in MACRO_COLS for lag in (1, 2, 3)),
        *(f"macro_stress_index_lag{lag}" for lag in (1, 2, 3)),
    ]

    for term in candidate_terms:
        model_data = hazard_base.dropna(subset=[term]).copy()
        groups = model_data["dt_referencia"].dt.to_period("M").astype(str)
        result = smf.glm(
            f"{HAZARD_BASE_FORMULA} + {term}", model_data, family=Binomial()
        ).fit(cov_type="cluster", cov_kwds={"groups": groups})
        conf = result.conf_int().loc[term]
        rows.append(
            {
                "macro_term": term,
                "coef": result.params[term],
                "std_error_cluster_month": result.bse[term],
                "p_value_cluster_month": result.pvalues[term],
                "odds_ratio": math.exp(result.params[term]),
                "or_ci_low": math.exp(conf[0]),
                "or_ci_high": math.exp(conf[1]),
                "aic": result.aic,
                "months": model_data["dt_referencia"].nunique(),
            }
        )
    return pd.DataFrame(rows).sort_values(["p_value_cluster_month", "aic"])


def make_contract_effects(result, contract_base: pd.DataFrame) -> pd.DataFrame:
    reference = contract_base.copy()
    effects = [
        average_probability_delta(
            result, reference, {"score_100": lambda s: s - 1}, "Score -100 pontos", "Originação"
        ),
        average_probability_delta(
            result, reference, {"ltv_10pp": lambda s: s + 1}, "LTV +10 p.p.", "Originação"
        ),
        average_probability_delta(
            result,
            reference,
            {"tx_juros_anual": lambda s: s + 1},
            "Taxa juros +1 p.p.",
            "Originação",
        ),
        average_probability_delta(
            result, reference, {"prazo_5anos": lambda s: s + 1}, "Prazo +5 anos", "Originação"
        ),
        average_probability_delta(
            result,
            reference,
            {"log_vr_financiado": lambda s: s + math.log(2)},
            "Valor financiado 2x",
            "Originação",
        ),
    ]

    for renda in ["Até 3 SM", "3 a 6 SM", "6 a 10 SM"]:
        effects.append(
            average_category_contrast(
                result,
                reference,
                "faixa_renda",
                "Acima de 10 SM",
                renda,
                f"Renda {renda} vs. >10 SM",
                "Renda",
            )
        )
    for programa in ["FGTS", "MCMV"]:
        effects.append(
            average_category_contrast(
                result,
                reference,
                "programa_social",
                "Livre",
                programa,
                f"Programa {programa} vs. Livre",
                "Programa",
            )
        )
    return pd.DataFrame(effects).sort_values("marginal_effect_pp", ascending=False)


def make_hazard_macro_effects(result, model_data: pd.DataFrame, macro_term: str) -> pd.DataFrame:
    p0 = result.predict(model_data)
    p1 = predict_with_changes(result, model_data, {macro_term: lambda s: s - 10})
    p2 = predict_with_changes(result, model_data, {"score_100": lambda s: s - 1})
    p3 = predict_with_changes(result, model_data, {"ltv_10pp": lambda s: s + 1})
    return pd.DataFrame(
        [
            {
                "factor": "Confianca consumidor -10 pontos",
                "marginal_effect_pp": (p1.mean() - p0.mean()) * 100,
                "base_probability_pct": p0.mean() * 100,
                "scenario_probability_pct": p1.mean() * 100,
            },
            {
                "factor": "Score -100 pontos",
                "marginal_effect_pp": (p2.mean() - p0.mean()) * 100,
                "base_probability_pct": p0.mean() * 100,
                "scenario_probability_pct": p2.mean() * 100,
            },
            {
                "factor": "LTV +10 p.p.",
                "marginal_effect_pp": (p3.mean() - p0.mean()) * 100,
                "base_probability_pct": p0.mean() * 100,
                "scenario_probability_pct": p3.mean() * 100,
            },
        ]
    )


def make_decile_calibration(y: pd.Series, pred: pd.Series, model_name: str) -> pd.DataFrame:
    out = pd.DataFrame({"y": y, "pred": pred})
    out["decile"] = pd.qcut(out["pred"], 10, labels=False, duplicates="drop") + 1
    return (
        out.groupby("decile", as_index=False)
        .agg(observed_rate=("y", "mean"), predicted_rate=("pred", "mean"), rows=("y", "size"))
        .assign(model=model_name)
    )
