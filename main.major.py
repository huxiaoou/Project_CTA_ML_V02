"""
0. 按品种生成主力合约连续数据, 以及价格复权数据
1. 保存路径为 /var/data/StackData/futures/team/huxo/major/au.ss

|          tp           | trading_day | ticker | preopen  | preclose |   open   |   high   |   low    |  close   |     vol     |    amount     |   oi   | major_return | major_return_open |  openM   |  highM   |   lowM   |  closeM  |
|-----------------------|-------------|--------|----------|----------|----------|----------|----------|----------|-------------|---------------|--------|--------------|-------------------|----------|----------|----------|----------|
| 20240318 16:00:00.000 |    20240318 | au2406 | 505.0800 | 508.1600 | 506.4600 | 508.1600 | 503.3600 | 504.0400 | 161567.0000 |  8174069.3504 | 224743 |      -0.0081 |            0.0027 | 413.2577 | 414.6449 | 410.7282 | 411.2831 |
| 20240319 16:00:00.000 |    20240319 | au2406 | 506.4600 | 504.0400 | 506.2000 | 506.7800 | 504.2800 | 505.2200 | 126762.0000 |  6410048.3072 | 223609 |       0.0023 |           -0.0005 | 413.0456 | 413.5188 | 411.4789 | 412.2459 |
| 20240320 16:00:00.000 |    20240320 | au2406 | 506.2000 | 505.2200 | 505.3800 | 506.3200 | 504.4400 | 505.7400 |  92479.0000 |  4674917.5808 | 224869 |       0.0010 |           -0.0016 | 412.3765 | 413.1435 | 411.6094 | 412.6702 |
| 20240321 16:00:00.000 |    20240321 | au2406 | 505.3800 | 505.7400 | 505.2400 | 516.0000 | 504.9000 | 515.7000 | 238295.0000 | 12171548.2624 | 234834 |       0.0197 |           -0.0003 | 412.2622 | 421.0421 | 411.9848 | 420.7973 |
| 20240322 16:00:00.000 |    20240322 | au2406 | 505.2400 | 515.7000 | 515.5000 | 515.9400 | 508.8200 | 512.4800 | 253868.0000 | 12992005.7344 | 228361 |      -0.0062 |            0.0203 | 420.6341 | 420.9931 | 415.1834 | 418.1699 |

"""

import argparse
import numpy as np


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To calculate major return and some other data")
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--nomp", default=False, action="store_true", help="not using multiprocess, for debug")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from hUtils.calendar import CCalendar
    from hUtils.instruments import CInstrumentInfoTable
    from hUtils.typeDef import TPatchData
    from hSolutions.sMajor import main_major
    from config import cfg_path, cfg_strategy

    basic_inputs = ["preclose", "open", "high", "low", "close", "vol", "amount", "oi"]
    args = parse_args()
    calendar = CCalendar(cfg_path.path_calendar)
    instru_info_tab = CInstrumentInfoTable(cfg_path.path_instru_info)

    # patch data
    # Special case, ni is not tradable at this date, due to continuous rise stop
    patch_data = TPatchData(
        {
            "ni": {
                (np.int32(20220310), b"ni2204"): {"open": 267700, "high": 267700, "low": 267700, "close": 267700},
                (np.int32(20220311), b"ni2204"): {"preopen": 267700},
            }
        }
    )

    main_major(
        universe=list(cfg_strategy.universe),
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        basic_inputs=basic_inputs,
        path_tsdb_futhot=cfg_path.path_tsdb_futhot,
        path_tsdb_fut=cfg_path.path_tsdb_fut,
        major_dir=cfg_path.major_dir,
        calendar=calendar,
        instru_info_tab=instru_info_tab,
        patch_data=patch_data,
        call_multiprocess=not args.nomp,
    )
