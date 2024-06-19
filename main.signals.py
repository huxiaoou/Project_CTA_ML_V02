import argparse
import numpy as np


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To calculate major return and some other data")
    arg_parser.add_argument(
        "--type", type=str, choices=("single", "portfolio"), help="single model or portfolios", required=True
    )
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--nomp", default=False, action="store_true", help="not using multiprocess, for debug")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from hUtils.calendar import CCalendar
    from hSolutions.sMclrn import load_config_models, get_tests, get_signal_args_ss
    from hSolutions.sSignals import main_translate_signals, main_translate_signals_portfolio
    from hSolutions.sSimulation import get_sim_args_from_tests, get_portfolio_args
    from config import cfg_path, PATH_CONFIG_MODELS, cfg_strategy

    calendar = CCalendar(cfg_path.path_calendar)
    args = parse_args()

    config_models = load_config_models(PATH_CONFIG_MODELS)
    tests = get_tests(config_models=config_models)

    if args.type == "single":
        signal_args = get_signal_args_ss(
            tests=tests, prediction_dir=cfg_path.prediction_dir, signals_dir=cfg_path.signals_dir
        )
        main_translate_signals(
            signal_args=signal_args,
            maw=cfg_strategy.CONST["MAW"],
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            calendar=calendar,
            call_multiprocess=not args.nomp,
        )
    elif args.type == "portfolio":
        sim_args = get_sim_args_from_tests(
            tests=tests, prefix_user=cfg_strategy.CONST["PREFIX_USER"], cost=cfg_strategy.CONST["COST"]
        )
        portfolio_args = get_portfolio_args(portfolios=cfg_strategy.portfolios, sim_args=sim_args)
        main_translate_signals_portfolio(
            portfolio_args=portfolio_args,
            tsdb_root_dir=cfg_path.path_tsdb_futhot,
            freq=cfg_strategy.CONST["FREQ"],
            prefix_user=cfg_strategy.CONST["PREFIX_USER"],
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            calendar=calendar,
            call_multiprocess=not args.nomp,
        )
    else:
        raise ValueError(f"argument 'type' = {args.type} is illegal")
