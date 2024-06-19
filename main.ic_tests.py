import argparse
import numpy as np


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To calculate major return and some other data")
    arg_parser.add_argument(
        "--switch", type=str, choices=("cal", "plt", "rol", "slc"), help="calculate or plot", required=True
    )
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--nomp", default=False, action="store_true", help="not using multiprocess, for debug")
    arg_parser.add_argument("--verbose", default=False, action="store_true", help="to print more details")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from hUtils.calendar import CCalendar
    from config import cfg_factors, cfg_path, cfg_strategy

    calendar = CCalendar(cfg_path.path_calendar)
    args = parse_args()

    if args.switch == "cal":
        from hSolutions.sICTests import main_cal_ic_tests

        main_cal_ic_tests(
            cfg_factors=list(cfg_factors.values()),
            ret_class=cfg_strategy.CONST["RET_CLASS"],
            ret_names=cfg_strategy.CONST["RET_NAMES"],
            shift=cfg_strategy.CONST["SHIFT"],
            tsdb_root_dir=cfg_path.path_tsdb_futhot,
            prefix_user=cfg_strategy.CONST["PREFIX_USER"],
            available_dir=cfg_path.available_dir,
            ic_save_dir=cfg_path.ic_tests_dir,
            freq=cfg_strategy.CONST["FREQ"],
            sectors=cfg_strategy.CONST["SECTORS"],
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            calendar=calendar,
            call_multiprocess=not args.nomp,
            processes=None,
            verbose=args.verbose,
        )
    elif args.switch == "plt":
        from hSolutions.sICTests import main_plt_ic_tests

        main_plt_ic_tests(
            cfg_factors=list(cfg_factors.values()),
            ret_names=cfg_strategy.CONST["RET_NAMES"],
            ic_save_dir=cfg_path.ic_tests_dir,
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            call_multiprocess=not args.nomp,
        )
    elif args.switch == "rol":
        from hSolutions.sICTests import main_rol_ic_tests

        main_rol_ic_tests(
            cfg_factors=list(cfg_factors.values()),
            rolling_wins=cfg_strategy.trn["wins"],
            ret_names=cfg_strategy.CONST["RET_NAMES"],
            shift=cfg_strategy.CONST["SHIFT"],
            sectors=cfg_strategy.CONST["SECTORS"],
            ic_save_dir=cfg_path.ic_tests_dir,
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            calendar=calendar,
            call_multiprocess=not args.nomp,
        )
    elif args.switch == "slc":
        from hSolutions.sICTests import main_sum_ic_tests

        main_sum_ic_tests(
            top_ratios=cfg_strategy.icir["top_ratios"],
            init_threshold=cfg_strategy.icir["init_threshold"],
            ret_class=cfg_strategy.CONST["RET_CLASS"],
            ret_names=cfg_strategy.CONST["RET_NAMES"],
            shift=cfg_strategy.CONST["SHIFT"],
            rolling_wins=cfg_strategy.trn["wins"],
            cfg_factors=list(cfg_factors.values()),
            sectors=cfg_strategy.CONST["SECTORS"],
            ic_save_dir=cfg_path.ic_tests_dir,
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            calendar=calendar,
            call_multiprocess=not args.nomp,
        )
    else:
        raise ValueError(f"[INF] argument switch = {args.switch} is illegal")
