"""
0. 另类数据: 汇率, CPI, M2, PPI
1. 保存路径为 /var/data/StackData/futures/team/huxo/alternative/

forex.ss:
|          tp           | trading_day | preclose |  open  |  high  |  low   | close  | pct_chg |
|-----------------------|-------------|----------|--------|--------|--------|--------|---------|
| 20240318 16:00:00.000 |    20240318 |   7.1965 | 7.1940 | 7.1986 | 7.1940 | 7.1982 |  0.0236 |
| 20240319 16:00:00.000 |    20240319 |   7.1982 | 7.1950 | 7.1997 | 7.1950 | 7.1993 |  0.0153 |
| 20240320 16:00:00.000 |    20240320 |   7.1993 | 7.1962 | 7.1998 | 7.1962 | 7.1993 |  0.0000 |
| 20240321 16:00:00.000 |    20240321 |   7.1993 | 7.1920 | 7.2002 | 7.1920 | 7.1994 |  0.0014 |
| 20240322 16:00:00.000 |    20240322 |   7.1994 | 7.1950 | 7.2304 | 7.1950 | 7.2283 |  0.4014 |

macro.ss
|          tp           | trading_day | cpi_rate | m2_rate | ppi_rate |
|-----------------------|-------------|----------|---------|----------|
| 20240318 16:00:00.000 |    20240318 |   -0.800 |   8.700 |   -2.500 |
| 20240319 16:00:00.000 |    20240319 |   -0.800 |   8.700 |   -2.500 |
| 20240320 16:00:00.000 |    20240320 |   -0.800 |   8.700 |   -2.500 |
| 20240321 16:00:00.000 |    20240321 |   -0.800 |   8.700 |   -2.500 |
| 20240322 16:00:00.000 |    20240322 |   -0.800 |   8.700 |   -2.500 |

"""

import argparse
import numpy as np

def parse_args():
    arg_parser = argparse.ArgumentParser(description="To calculate alternative data, such as macro and forex")
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from config import cfg_path
    from hUtils.calendar import CCalendar
    from hSolutions.sAlternative import main_macro, main_forex

    calendar = CCalendar(cfg_path.path_calendar)
    args = parse_args()
    main_macro(
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        path_macro_data=cfg_path.path_macro_data,
        alternative_dir=cfg_path.alternative_dir,
        calendar=calendar,
    )
    main_forex(
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        path_forex_data=cfg_path.path_forex_data,
        alternative_dir=cfg_path.alternative_dir,
        calendar=calendar,
    )
