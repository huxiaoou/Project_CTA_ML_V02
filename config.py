import yaml
import dataclasses

PATH_CONFIG = "/home/huxo/Research/Projects/Project_CTA_ML_V02/config.yaml"
PATH_CONFIG_MODELS = "/home/huxo/Research/Projects/Project_CTA_ML_V02/config_models.yaml"


@dataclasses.dataclass(frozen=True)
class ConfigPath:
    path_stack_data_fut: str
    path_tsdb_fut: str
    path_tsdb_futhot: str
    path_calendar: str
    path_instru_info: str
    path_basis_and_fundamental: str
    path_macro_data: str
    path_forex_data: str
    path_spot_data: str
    path_mkt_idx_data: str

    project_data_save_dir: str
    minute_bar_dir: str
    alternative_dir: str
    mbr_pos_dir: str
    preprocess_dir: str
    major_dir: str
    available_dir: str
    market_dir: str

    factors_by_instru_dir: str
    neutral_by_instru_dir: str

    y_dir: str
    feature_selection_dir: str
    mclrn_dir: str
    prediction_dir: str
    signals_dir: str
    simulations_dir: str
    evaluations_dir: str


@dataclasses.dataclass(frozen=True)
class ConfigStrategy:
    account_wds: dict[str, str]
    mkt_idxes: dict[str, str]
    universe: dict[str, dict[str, str]]
    available_universe: dict
    factors: dict[str, dict]
    trn: dict
    feature_selection: dict
    mclrns: dict
    portfolios: dict[str, dict]
    CONST: dict


with open(PATH_CONFIG, "r") as f:
    from hSolutions.sFactorAlg import (
        CCfgFactors,
        CCfgFactorMTM,
        CCfgFactorSKEW,
        CCfgFactorRS,
        CCfgFactorBASIS,
        CCfgFactorTS,
        CCfgFactorS0BETA,
        CCfgFactorS1BETA,
        CCfgFactorCBETA,
        CCfgFactorIBETA,
        CCfgFactorPBETA,
        CCfgFactorCTP,
        CCfgFactorCTR,
        CCfgFactorCVP,
        CCfgFactorCVR,
        CCfgFactorCSP,
        CCfgFactorCSR,
        CCfgFactorNOI,
        CCfgFactorNDOI,
        CCfgFactorWNOI,
        CCfgFactorWNDOI,
        CCfgFactorAMP,
        CCfgFactorEXR,
        CCfgFactorSMT,
        CCfgFactorRWTC,
    )

    __cfg_path_data, __cfg_strategy_data = yaml.safe_load_all(f)
    cfg_path, cfg_strategy = ConfigPath(**__cfg_path_data), ConfigStrategy(**__cfg_strategy_data)
    cfg_factors = CCfgFactors(
        MTM=CCfgFactorMTM(**cfg_strategy.factors["MTM"]),
        SKEW=CCfgFactorSKEW(**cfg_strategy.factors["SKEW"]),
        RS=CCfgFactorRS(**cfg_strategy.factors["RS"]),
        BASIS=CCfgFactorBASIS(**cfg_strategy.factors["BASIS"]),
        TS=CCfgFactorTS(**cfg_strategy.factors["TS"]),
        S0BETA=CCfgFactorS0BETA(**cfg_strategy.factors["S0BETA"]),
        S1BETA=CCfgFactorS1BETA(**cfg_strategy.factors["S1BETA"]),
        CBETA=CCfgFactorCBETA(**cfg_strategy.factors["CBETA"]),
        IBETA=CCfgFactorIBETA(**cfg_strategy.factors["IBETA"]),
        PBETA=CCfgFactorPBETA(**cfg_strategy.factors["PBETA"]),
        CTP=CCfgFactorCTP(**cfg_strategy.factors["CTP"]),
        CTR=None,  # CCfgFactorCTR(**cfg_strategy.factors["CTR"]),
        CVP=CCfgFactorCVP(**cfg_strategy.factors["CVP"]),
        CVR=None,  # CCfgFactorCVR(**cfg_strategy.factors["CVR"]),
        CSP=CCfgFactorCSP(**cfg_strategy.factors["CSP"]),
        CSR=None,  # CCfgFactorCSR(**cfg_strategy.factors["CSR"]),
        NOI=CCfgFactorNOI(**cfg_strategy.factors["NOI"]),
        NDOI=CCfgFactorNDOI(**cfg_strategy.factors["NDOI"]),
        WNOI=None,  # CCfgFactorWNOI(**cfg_strategy.factors["WNOI"]),
        WNDOI=None,  # CCfgFactorWNDOI(**cfg_strategy.factors["WNDOI"]),
        AMP=CCfgFactorAMP(**cfg_strategy.factors["AMP"]),
        EXR=CCfgFactorEXR(**cfg_strategy.factors["EXR"]),
        SMT=CCfgFactorSMT(**cfg_strategy.factors["SMT"]),
        RWTC=CCfgFactorRWTC(**cfg_strategy.factors["RWTC"]),
    )

    factors_pool_raw, factors_pool_neu = [], []
    for cfg_factor in cfg_factors.values():
        for factor_class, factor_names, _ in cfg_factor.get_combs_raw():
            factors_pool_raw.extend([(factor_class, n) for n in factor_names])
        for factor_class, factor_names, _ in cfg_factor.get_combs_neu():
            factors_pool_neu.extend([(factor_class, n) for n in factor_names])

if __name__ == "__main__":
    import pandas as pd
    from hUtils.tools import check_and_mkdir

    check_and_mkdir(cfg_path.project_data_save_dir, True)
    check_and_mkdir(cfg_path.alternative_dir, True)
    check_and_mkdir(cfg_path.mbr_pos_dir, True)
    check_and_mkdir(cfg_path.preprocess_dir, True)
    check_and_mkdir(cfg_path.major_dir, True)
    check_and_mkdir(cfg_path.available_dir, True)
    check_and_mkdir(cfg_path.market_dir, True)
    check_and_mkdir(cfg_path.factors_by_instru_dir, True)
    check_and_mkdir(cfg_path.neutral_by_instru_dir, True)
    check_and_mkdir(cfg_path.y_dir, True)
    check_and_mkdir(cfg_path.feature_selection_dir, True)
    check_and_mkdir(cfg_path.mclrn_dir, True)
    check_and_mkdir(cfg_path.prediction_dir, True)
    check_and_mkdir(cfg_path.signals_dir, True)
    check_and_mkdir(cfg_path.simulations_dir, True)
    check_and_mkdir(cfg_path.evaluations_dir, True)

    print(f"[INF] There {len(cfg_strategy.CONST['SECTORS'])} sectors, they are: {cfg_strategy.CONST['SECTORS']}")
    sector = pd.DataFrame.from_dict(cfg_strategy.universe, orient="index")
    for sector_l1, sector_data in sector.groupby(by="sectorL1"):
        print(f"[INF] Sector {sector_l1}: size = {len(sector_data):>2d}, contains: {sector_data.index.tolist()}")
    print(f"[INF] Size of universe = {len(cfg_strategy.universe)}")

    print(f"[INF] Quantity of factors before neutralization = {len(factors_pool_raw)}")
    print(f"[INF] Quantity of factors after  neutralization = {len(factors_pool_neu)}")
