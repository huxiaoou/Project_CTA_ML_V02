"""
0. 计算成交额加权的市场收益, 行业收益.
1. 保存路径 /var/data/StackData/futures/team/huxo/market/market.ss

|          tp           | trading_day | market  |    C    |    F    |   GLD   |   MTL   |   OIL   |   CHM   |   BLK   |   AGR   |   EQT   |
|-----------------------|-------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 20240311 16:00:00.000 |    20240311 | -0.0058 | -0.0093 |  0.0171 |  0.0020 | -0.0083 | -0.0048 | -0.0040 | -0.0271 | -0.0063 |  0.0171 |
| 20240312 16:00:00.000 |    20240312 |  0.0037 |  0.0041 |  0.0012 | -0.0020 |  0.0075 |  0.0095 | -0.0015 |  0.0047 |  0.0061 |  0.0012 |
| 20240313 16:00:00.000 |    20240313 | -0.0031 | -0.0033 | -0.0014 | -0.0019 |  0.0055 |  0.0004 |  0.0005 | -0.0157 | -0.0065 | -0.0014 |
| 20240314 16:00:00.000 |    20240314 |  0.0019 |  0.0026 | -0.0039 |  0.0125 |  0.0021 |  0.0148 |  0.0112 | -0.0155 | -0.0078 | -0.0039 |
| 20240315 16:00:00.000 |    20240315 |  0.0061 |  0.0056 |  0.0106 |  0.0063 |  0.0113 |  0.0032 |  0.0119 | -0.0039 |  0.0044 |  0.0106 |

"""

import argparse
import numpy as np


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="To calculate market and sector return, must be run after main.available.py"
    )
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from hUtils.calendar import CCalendar
    from hSolutions.sMarket import main_market
    from config import cfg_path, cfg_strategy

    calendar = CCalendar(cfg_path.path_calendar)
    args = parse_args()

    main_market(
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        calendar=calendar,
        available_dir=cfg_path.available_dir,
        market_dir=cfg_path.market_dir,
        path_mkt_idx_data=cfg_path.path_mkt_idx_data,
        mkt_idxes=list(cfg_strategy.mkt_idxes.values())
    )
