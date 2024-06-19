import argparse
import numpy as np


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To translate minute bar in tsdb to '.ss' file")
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    arg_parser.add_argument(
        "--freq", type=str, default="m01e", choices=("m01e", "m05e", "m15e", "d01e"), help="frequency to load data"
    )
    arg_parser.add_argument("--nomp", default=False, action="store_true", help="not using multiprocess, for debug")
    arg_parser.add_argument("--verbose", default=False, action="store_true", help="to print more details")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from hUtils.calendar import CCalendar
    from hSolutions.sMinBar import main_min_bar
    from config import cfg_path

    VALUES = {
        "preclose": np.float32,
        "open": np.float32,
        "high": np.float32,
        "low": np.float32,
        "close": np.float32,
        "oi": np.float32,
        "vol": np.float32,
        "amount": np.float32,
        "daily_open": np.float32,
        "daily_high": np.float32,
        "daily_low": np.float32,
    }
    VARS_TO_DROP_INVALID = ["open", "high", "low", "close", "amount"]
    BATCH_SIZE = 250
    calendar = CCalendar(calendar_path=cfg_path.path_calendar)
    args = parse_args()

    main_min_bar(
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        tsdb_root_dir=cfg_path.path_tsdb_futhot,
        freq=args.freq,
        values=VALUES,
        vars_to_drop_invalid=VARS_TO_DROP_INVALID,
        minute_bar_dir=cfg_path.minute_bar_dir,
        calendar=calendar,
        batch_size=BATCH_SIZE,
    )
