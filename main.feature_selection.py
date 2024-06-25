import argparse
import numpy as np


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To calculate major return and some other data")
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--nomp", default=False, action="store_true", help="not using multiprocess, for debug")
    arg_parser.add_argument("--processes", type=int, default=12, help="Number of processes to be called")
    arg_parser.add_argument("--verbose", default=False, action="store_true", help="print more information")
    return arg_parser.parse_args()


if __name__ == "__main__":
    import os
    from hUtils.typeDef import CRet
    from hUtils.calendar import CCalendar
    from hSolutions.sFeatureSelection import main_feature_selection, get_feature_selection_tests
    from config import cfg_path, cfg_strategy, factors_pool_neu

    calendar = CCalendar(cfg_path.path_calendar)
    args = parse_args()

    rets = [
        CRet(
            ret_class=cfg_strategy.CONST["RET_CLASS"],
            ret_name=n,
            shift=cfg_strategy.CONST["SHIFT"],
        )
        for n in cfg_strategy.CONST["RET_NAMES"]
    ]
    tests = get_feature_selection_tests(
        trn_wins=cfg_strategy.trn["wins"],
        sectors=cfg_strategy.CONST["SECTORS"],
        rets=rets,
    )
    main_feature_selection(
        threshold=cfg_strategy.feature_selection["mut_info_threshold"],
        min_feats=cfg_strategy.feature_selection["min_feats"],
        tests=tests,
        facs_pool=factors_pool_neu,
        tsdb_root_dir=cfg_path.path_tsdb_futhot,
        tsdb_user_prefix=cfg_strategy.CONST["PREFIX_USER"],
        freq=cfg_strategy.CONST["FREQ"],
        avlb_path=os.path.join(cfg_path.available_dir, "available.ss"),
        feature_selection_dir=cfg_path.feature_selection_dir,
        universe=cfg_strategy.universe,
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        calendar=calendar,
        call_multiprocess=not args.nomp,
        processes=args.processes,
        verbose=args.verbose,
    )
