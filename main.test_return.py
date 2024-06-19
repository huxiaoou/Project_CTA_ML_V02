import numpy as np
import argparse


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To calculate factors")
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--nomp", default=False, action="store_true", help="not using multiprocess, for debug")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from config import cfg_strategy, cfg_path
    from hUtils.calendar import CCalendar
    from hSolutions.sTestReturn import CTstRet, CTstRetNeu

    calendar = CCalendar(cfg_path.path_calendar)
    args = parse_args()
    PROCESSES = 12

    for lag, win in zip((cfg_strategy.CONST["LAG"], 1), (cfg_strategy.CONST["WIN"], 1)):
        # test return
        tst_ret = CTstRet(
            lag=lag,
            win=win,
            universe=list(cfg_strategy.universe),
            major_dir=cfg_path.major_dir,
            save_root_dir=cfg_path.y_dir,
        )
        tst_ret.main_test_return(
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            calendar=calendar,
        )

        # neutralization
        tst_ret_neu = CTstRetNeu(
            lag=lag,
            win=win,
            universe=list(cfg_strategy.universe),
            major_dir=cfg_path.major_dir,
            available_dir=cfg_path.available_dir,
            save_root_dir=cfg_path.y_dir,
        )
        tst_ret_neu.main_test_return_neutralize(
            bgn_date=args.bgn,
            end_date=args.end or args.bgn,
            calendar=calendar,
            call_multiprocess=not args.nomp,
            processes=PROCESSES,
        )
