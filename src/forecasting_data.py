from __future__ import annotations

import pandas as pd

from inference_config import DATA_PATH


def load_monthly_delinquency_series() -> pd.DataFrame:
    parcelas = pd.read_excel(DATA_PATH, sheet_name="parcelas")
    contratos = pd.read_excel(DATA_PATH, sheet_name="contratos")
    macro = pd.read_excel(DATA_PATH, sheet_name="indicadores_macro")
    parcelas["dt_referencia"] = pd.to_datetime(parcelas["dt_referencia"])
    macro["dt_referencia"] = pd.to_datetime(macro["dt_referencia"])

    panel = parcelas.merge(contratos, on="id_contrato", how="left")
    panel["early_arrears"] = panel["dias_atraso"].between(1, 90).astype(int)
    panel["arrears_1_30"] = panel["dias_atraso"].between(1, 30).astype(int)
    panel["arrears_31_60"] = panel["dias_atraso"].between(31, 60).astype(int)
    panel["arrears_61_90"] = panel["dias_atraso"].between(61, 90).astype(int)
    panel["low_income"] = panel["faixa_renda"].eq("Até 3 SM").astype(int)
    panel["social_program"] = panel["programa_social"].ne("Livre").astype(int)

    monthly = (
        panel.groupby("dt_referencia", as_index=False)
        .agg(
            contratos_ativos=("id_contrato", "nunique"),
            contratos_inadimplentes=("fl_inadimplente", "sum"),
            taxa_inadimplencia=("fl_inadimplente", "mean"),
            vr_parcela_devida_total=("vr_parcela_devida", "sum"),
            vr_pago_total=("vr_pago", "sum"),
            dias_atraso_medio=("dias_atraso", "mean"),
            saldo_devedor_medio=("saldo_devedor", "mean"),
            share_atraso_1_30=("arrears_1_30", "mean"),
            share_atraso_31_60=("arrears_31_60", "mean"),
            share_atraso_61_90=("arrears_61_90", "mean"),
            share_atraso_1_90=("early_arrears", "mean"),
            ltv_medio=("ltv", "mean"),
            score_medio=("score_credito_contratacao", "mean"),
            share_baixa_renda=("low_income", "mean"),
            share_programa_social=("social_program", "mean"),
        )
        .sort_values("dt_referencia")
    )
    monthly["taxa_inadimplencia_pct"] = monthly["taxa_inadimplencia"] * 100
    monthly["pct_pago_devido"] = monthly["vr_pago_total"] / monthly["vr_parcela_devida_total"] * 100
    share_cols = [
        "share_atraso_1_30",
        "share_atraso_31_60",
        "share_atraso_61_90",
        "share_atraso_1_90",
        "share_baixa_renda",
        "share_programa_social",
    ]
    monthly[share_cols] = monthly[share_cols] * 100
    monthly = monthly.merge(macro, on="dt_referencia", how="left")
    return monthly


def split_train_validation(
    monthly: pd.DataFrame, validation_start: str = "2024-07-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(validation_start)
    train = monthly.loc[monthly["dt_referencia"] < cutoff].copy()
    validation = monthly.loc[monthly["dt_referencia"] >= cutoff].copy()
    return train, validation


def make_future_months(start: str = "2025-01-01", periods: int = 6) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="MS")
