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
    from hSolutions.sMclrn import load_config_models, get_tests
    from hSolutions.sSimulation import main_simulations
    from config import cfg_path, cfg_strategy, PATH_CONFIG_MODELS

    calendar = CCalendar(cfg_path.path_calendar)
    args = parse_args()

    if args.type == "single":
        from hSolutions.sSimulation import get_sim_args_from_tests

        config_models = load_config_models(PATH_CONFIG_MODELS)
        tests = get_tests(config_models=config_models)
        simu_args = get_sim_args_from_tests(
            tests=tests,
            prefix_user=cfg_strategy.CONST["PREFIX_USER"],
            cost=cfg_strategy.CONST["COST"],
        )
        main_simulations(
            sim_args=simu_args,
            tsdb_root_dir=cfg_path.path_tsdb_futhot,
            freq=cfg_strategy.CONST["FREQ"],
            sim_save_dir=cfg_path.simulations_dir,
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            calendar=calendar,
            call_multiprocess=not args.nomp,
        )
    elif args.type == "portfolio":
        from hSolutions.sSimulation import get_sim_args_from_portfolios

        simu_args = get_sim_args_from_portfolios(
            portfolios=cfg_strategy.portfolios,
            prefix_user=cfg_strategy.CONST["PREFIX_USER"],
            cost=cfg_strategy.CONST["COST"],
            shift=cfg_strategy.CONST["SHIFT"],
        )
        main_simulations(
            sim_args=simu_args,
            tsdb_root_dir=cfg_path.path_tsdb_futhot,
            freq=cfg_strategy.CONST["FREQ"],
            sim_save_dir=cfg_path.simulations_dir,
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            calendar=calendar,
            call_multiprocess=not args.nomp,
        )
    else:
        raise ValueError(f"argument 'type' = {args.type} is illegal")
