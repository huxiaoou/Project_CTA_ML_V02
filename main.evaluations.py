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
    from hSolutions.sSimulation import get_sim_args_from_tests
    from hSolutions.sMclrnManage import load_config_models, get_tests
    from config import cfg_path, cfg_strategy, PATH_CONFIG_MODELS

    calendar = CCalendar(cfg_path.path_calendar)
    args = parse_args()

    if args.type == "single":
        from hSolutions.sEvaluation import main_eval_tests, main_plot_sims, main_plot_sims_by_sector

        config_models = load_config_models(PATH_CONFIG_MODELS)
        tests = get_tests(config_models=config_models)
        sim_args = get_sim_args_from_tests(
            tests=tests,
            prefix_user=cfg_strategy.CONST["PREFIX_USER"],
            cost=cfg_strategy.CONST["COST"],
        )
        main_eval_tests(
            sim_args=sim_args,
            simulations_dir=cfg_path.simulations_dir,
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            call_multiprocess=not args.nomp,
        )
        main_plot_sims(
            sim_args=sim_args,
            simulations_dir=cfg_path.simulations_dir,
            evaluations_dir=cfg_path.evaluations_dir,
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            call_multiprocess=not args.nomp,
        )
        main_plot_sims_by_sector(
            sim_args=sim_args,
            simulations_dir=cfg_path.simulations_dir,
            evaluations_dir=cfg_path.evaluations_dir,
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            call_multiprocess=not args.nomp,
        )
    elif args.type == "portfolio":
        from hSolutions.sEvaluation import main_eval_portfolios, main_plot_portfolios

        main_eval_portfolios(
            portfolios=cfg_strategy.portfolios,
            simulations_dir=cfg_path.simulations_dir,
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            call_multiprocess=not args.nomp,
        )
        main_plot_portfolios(
            portfolios=cfg_strategy.portfolios,
            simulations_dir=cfg_path.simulations_dir,
            evaluations_dir=cfg_path.evaluations_dir,
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
        )
