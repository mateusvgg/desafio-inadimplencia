from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "desafio_inadimplencia_caixa.xlsx"
OUTPUT_TABLES = ROOT / "outputs" / "tables"
OUTPUT_FIGURES = ROOT / "outputs" / "figures"

UF_TO_REGION = {
    **dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "Norte"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "Centro-Oeste"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
    **dict.fromkeys(["PR", "RS", "SC"], "Sul"),
}

MACRO_COLS = [
    "selic_12m",
    "ipca_12m",
    "tx_desemprego_12m",
    "rendimento_medio",
    "confianca_consumidor",
]
