from __future__ import annotations

import pandas as pd
import seaborn as sns

from inference_config import OUTPUT_FIGURES, OUTPUT_TABLES
from inference_data import (
    build_contract_base,
    build_hazard_base,
    count_left_censored_contracts,
    load_data,
)
from inference_modeling import (
    make_contract_effects,
    make_decile_calibration,
    make_hazard_macro_effects,
    run_contract_model,
    run_hazard_model,
    screen_macro_terms,
    tidy_glm_result,
)
from inference_plots import (
    save_calibration_plot,
    save_contract_effect_plot,
    save_hazard_effect_plot,
    save_odds_ratio_plot,
    save_risk_scenario_plot,
)


FINAL_MACRO_TERM = "confianca_consumidor_lag2"


def save_tables(
    contract_base: pd.DataFrame,
    hazard_model_data: pd.DataFrame,
    contract_results: pd.DataFrame,
    contract_effects: pd.DataFrame,
    contract_calibration: pd.DataFrame,
    macro_screen: pd.DataFrame,
    hazard_results: pd.DataFrame,
    hazard_effects: pd.DataFrame,
    scenarios: pd.DataFrame,
    diagnostics: list[dict],
) -> None:
    contract_base.to_csv(OUTPUT_TABLES / "inference_contract_base.csv", index=False)
    hazard_model_data.to_csv(OUTPUT_TABLES / "inference_hazard_model_base.csv", index=False)
    contract_results.to_csv(OUTPUT_TABLES / "inference_contract_model_results.csv", index=False)
    contract_effects.to_csv(OUTPUT_TABLES / "inference_contract_marginal_effects.csv", index=False)
    contract_calibration.to_csv(OUTPUT_TABLES / "inference_contract_calibration.csv", index=False)
    macro_screen.to_csv(OUTPUT_TABLES / "inference_macro_screening.csv", index=False)
    hazard_results.to_csv(OUTPUT_TABLES / "inference_hazard_model_results.csv", index=False)
    hazard_effects.to_csv(OUTPUT_TABLES / "inference_hazard_marginal_effects.csv", index=False)
    scenarios.to_csv(OUTPUT_TABLES / "inference_risk_scenarios.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        OUTPUT_TABLES / "inference_model_diagnostics.csv", index=False
    )


def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    contratos, parcelas, macro = load_data()
    contract_base = build_contract_base(contratos, parcelas)
    hazard_base = build_hazard_base(contratos, parcelas, macro)
    excluded_left_censored_contracts = count_left_censored_contracts(parcelas)

    # Main inference layer: origination characteristics explaining whether a
    # contract ever becomes delinquent during the observed window.
    contract_result, contract_pred, contract_diag = run_contract_model(contract_base)
    contract_results = tidy_glm_result(contract_result, "contract_logit_ever_delinquent")
    contract_effects = make_contract_effects(contract_result, contract_base)
    contract_calibration = make_decile_calibration(
        contract_base["fl_ever_inadimplente"],
        contract_pred,
        "contract_logit_ever_delinquent",
    )

    # Complementary macro layer: first monthly entry into delinquency, with
    # macro candidates screened before fitting the final interpretable term.
    macro_screen = screen_macro_terms(hazard_base)
    hazard_result, hazard_model_data, hazard_pred, hazard_diag = run_hazard_model(
        hazard_base, FINAL_MACRO_TERM, excluded_left_censored_contracts
    )
    hazard_results = tidy_glm_result(
        hazard_result, f"hazard_logit_entry_{FINAL_MACRO_TERM}"
    )
    hazard_effects = make_hazard_macro_effects(
        hazard_result, hazard_model_data, FINAL_MACRO_TERM
    )

    scenarios = save_risk_scenario_plot(contract_result, contract_base)
    save_contract_effect_plot(contract_effects)
    save_odds_ratio_plot(contract_results)
    save_calibration_plot(contract_calibration)
    save_hazard_effect_plot(hazard_effects)

    save_tables(
        contract_base,
        hazard_model_data,
        contract_results,
        contract_effects,
        contract_calibration,
        macro_screen,
        hazard_results,
        hazard_effects,
        scenarios,
        [contract_diag, hazard_diag],
    )

    print("Contract model diagnostics:", contract_diag)
    print("Hazard model diagnostics:", hazard_diag)
    print("Top contract marginal effects:")
    print(contract_effects.head(8).to_string(index=False))
    print("Macro screening top rows:")
    print(macro_screen.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
