from hSolutions.sMclrn import parse_model_configs

if __name__ == "__main__":
    from config import cfg_strategy, PATH_CONFIG_MODELS

    parse_model_configs(
        models=cfg_strategy.mclrns["models"],
        ret_class=cfg_strategy.CONST["RET_CLASS"],
        ret_names=cfg_strategy.CONST["RET_NAMES"],
        shift=cfg_strategy.CONST["SHIFT"],
        sectors=cfg_strategy.CONST["SECTORS"],
        trn_wins=cfg_strategy.trn["wins"],
        cfg_strategy_icir=cfg_strategy.icir,
        path_config_models=PATH_CONFIG_MODELS,
    )
