"""
0. 依据主力合约表现, 逐日确定成交活跃的品种, 并标注所属行业。
1. 保存路径为 /var/data/StackData/futures/team/huxo/available/available.ss

|          tp           | trading_day | instrument | return  |   amount   | sectorL0 | sectorL1 |
|-----------------------|-------------|------------|---------|------------|----------|----------|
| 20240322 16:00:00.000 |    20240322 |         cs |  0.0000 | 292680.344 |        C |      AGR |
| 20240322 16:00:00.000 |    20240322 |         PK |  0.0030 | 262071.625 |        C |      AGR |
| 20240322 16:00:00.000 |    20240322 |         PF | -0.0005 | 238221.828 |        C |      CHM |
| 20240322 16:00:00.000 |    20240322 |         br | -0.0055 | 223721.703 |        C |      CHM |
| 20240322 16:00:00.000 |    20240322 |         CJ | -0.0113 | 132741.938 |        C |      AGR |

"""

import argparse
import numpy as np


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To calculate available universe, must be run after main.major.py")
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from hUtils.calendar import CCalendar
    from hSolutions.sAvailable import CCfgAvlbUnvrs, main_available
    from config import cfg_path, cfg_strategy

    calendar = CCalendar(cfg_path.path_calendar)
    cfg_avlb_unvrs = CCfgAvlbUnvrs(
        universe=cfg_strategy.universe,
        win=cfg_strategy.available_universe["win"],
        amount_threshold=cfg_strategy.available_universe["amount_threshold"],
    )
    args = parse_args()
    main_available(
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        cfg_avlb_unvrs=cfg_avlb_unvrs,
        available_dir=cfg_path.available_dir,
        major_dir=cfg_path.major_dir,
        calendar=calendar,
    )
