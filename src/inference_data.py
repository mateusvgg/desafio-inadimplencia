from __future__ import annotations

import numpy as np
import pandas as pd

from inference_config import DATA_PATH, MACRO_COLS, UF_TO_REGION


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contratos = pd.read_excel(DATA_PATH, sheet_name="contratos")
    parcelas = pd.read_excel(DATA_PATH, sheet_name="parcelas")
    macro = pd.read_excel(DATA_PATH, sheet_name="indicadores_macro")

    for df in (contratos, parcelas, macro):
        for col in df.columns:
            if col.startswith("dt_"):
                df[col] = pd.to_datetime(df[col])

    return contratos, parcelas, macro


def add_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create scaled variables whose coefficients are easy to explain."""
    out = df.copy()
    out["regiao"] = out["uf"].map(UF_TO_REGION).fillna("Outras")
    out["score_100"] = out["score_credito_contratacao"] / 100
    out["ltv_10pp"] = out["ltv"] / 10
    out["prazo_5anos"] = out["prazo_meses"] / 60
    out["log_vr_financiado"] = np.log(out["vr_financiado"])
    return out


def build_contract_base(contratos: pd.DataFrame, parcelas: pd.DataFrame) -> pd.DataFrame:
    target = (
        parcelas.groupby("id_contrato", as_index=False)
        .agg(
            fl_ever_inadimplente=("fl_inadimplente", "max"),
            primeiro_mes_observado=("dt_referencia", "min"),
            ultimo_mes_observado=("dt_referencia", "max"),
            qtd_meses_observados=("dt_referencia", "nunique"),
            max_dias_atraso=("dias_atraso", "max"),
        )
    )
    base = contratos.merge(target, on="id_contrato", how="left")
    base["fl_ever_inadimplente"] = base["fl_ever_inadimplente"].fillna(0).astype(int)
    return add_contract_features(base)


def build_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    out = macro.sort_values("dt_referencia").copy()

    # A compact stress index is screened as a sensitivity option. It keeps the
    # macro layer interpretable and avoids overfitting five correlated monthly
    # series in a short 36-month history.
    stress_sign = {
        "selic_12m": 1,
        "ipca_12m": 1,
        "tx_desemprego_12m": 1,
        "rendimento_medio": -1,
        "confianca_consumidor": -1,
    }
    out["macro_stress_index"] = 0.0
    for col, sign in stress_sign.items():
        out["macro_stress_index"] += sign * (out[col] - out[col].mean()) / out[col].std(ddof=0)
    out["macro_stress_index"] = out["macro_stress_index"] / len(stress_sign)

    for col in MACRO_COLS + ["macro_stress_index"]:
        for lag in (1, 2, 3):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)

    return out


def build_hazard_base(
    contratos: pd.DataFrame, parcelas: pd.DataFrame, macro: pd.DataFrame
) -> pd.DataFrame:
    parcelas_ord = parcelas.sort_values(["id_contrato", "dt_referencia"]).copy()
    parcelas_ord["ever_prior"] = (
        parcelas_ord.groupby("id_contrato")["fl_inadimplente"]
        .cummax()
        .groupby(parcelas_ord["id_contrato"])
        .shift(fill_value=0)
    )
    first_status = (
        parcelas_ord.groupby("id_contrato")["fl_inadimplente"]
        .first()
        .rename("first_observed_inadimplente")
    )

    # Keep each contract-month only until the first delinquency event. Contracts
    # already delinquent at their first observed month are removed from the entry
    # model because the true entry month is unknown.
    panel = parcelas_ord.merge(first_status, on="id_contrato", how="left")
    panel = panel.loc[panel["first_observed_inadimplente"].eq(0)].copy()
    panel["entrada_inadimplencia"] = (
        panel["fl_inadimplente"].eq(1) & panel["ever_prior"].eq(0)
    ).astype(int)
    panel = panel.loc[panel["ever_prior"].eq(0) | panel["entrada_inadimplencia"].eq(1)].copy()

    panel = panel.merge(add_contract_features(contratos), on="id_contrato", how="left")
    panel = panel.merge(build_macro_features(macro), on="dt_referencia", how="left")
    panel["meses_desde_contratacao"] = (
        (panel["dt_referencia"].dt.year - panel["dt_contratacao"].dt.year) * 12
        + (panel["dt_referencia"].dt.month - panel["dt_contratacao"].dt.month)
    ).clip(lower=0)
    return panel


def count_left_censored_contracts(parcelas: pd.DataFrame) -> int:
    first_observed = (
        parcelas.sort_values(["id_contrato", "dt_referencia"])
        .groupby("id_contrato")["fl_inadimplente"]
        .first()
    )
    return int(first_observed.sum())
