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
    from hUtils.calendar import CCalendar
    from hSolutions.sMclrnManage import load_config_models, get_tests
    from hSolutions.sMclrn import main_train_and_predict
    from config import cfg_path, cfg_strategy, PATH_CONFIG_MODELS

    calendar = CCalendar(cfg_path.path_calendar)
    args = parse_args()

    config_models = load_config_models(PATH_CONFIG_MODELS)
    tests = get_tests(config_models=config_models)
    main_train_and_predict(
        tests=tests,
        tsdb_root_dir=cfg_path.path_tsdb_futhot,
        tsdb_user_prefix=cfg_strategy.CONST["PREFIX_USER"],
        freq=cfg_strategy.CONST["FREQ"],
        avlb_path=os.path.join(cfg_path.available_dir, "available.ss"),
        mclrn_dir=cfg_path.mclrn_dir,
        prediction_dir=cfg_path.prediction_dir,
        feature_selection_dir=cfg_path.feature_selection_dir,
        universe=cfg_strategy.universe,
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        calendar=calendar,
        call_multiprocess=not args.nomp,
        processes=args.processes,
        verbose=args.verbose,
    )
