"""
0. 按品种生成主力(近月)合约、次主力(远月)合约, 以及基差,仓单等数据
1. 保存路径为 /var/data/StackData/futures/team/huxo/preprocess/IC.ss

... 近月与远月合约

    |          tp           | trading_day | ticker_major | ticker_minor |
    |-----------------------|-------------|--------------|--------------|
    | 20240318 16:00:00.000 |    20240318 |       IC2404 |       IC2406 |
    | 20240319 16:00:00.000 |    20240319 |       IC2404 |       IC2406 |
    | 20240320 16:00:00.000 |    20240320 |       IC2404 |       IC2406 |
    | 20240321 16:00:00.000 |    20240321 |       IC2404 |       IC2406 |
    | 20240322 16:00:00.000 |    20240322 |       IC2404 |       IC2406 |

... 近月合约日行情

    | preclose_major | open_major | high_major | low_major | close_major | vol_major  | amount_major | oi_major |
    |----------------|------------|------------|-----------|-------------|------------|--------------|----------|
    |      5452.6001 |  5465.0000 |  5536.7998 | 5446.2002 |   5533.0000 | 43646.0000 | 4786680.6272 |    84233 |
    |      5533.0000 |  5517.2002 |  5525.6001 | 5465.0000 |   5472.2002 | 42311.0000 | 4654020.1984 |    84591 |
    |      5472.2002 |  5470.6001 |  5505.0000 | 5452.0000 |   5486.0000 | 41659.0000 | 4564160.9216 |    80698 |
    |      5486.0000 |  5496.7998 |  5512.7998 | 5446.3999 |   5457.7998 | 41773.0000 | 4572968.9600 |    82577 |
    |      5457.7998 |  5448.0000 |  5458.7998 | 5343.0000 |   5367.0000 | 56065.0000 | 6040006.6560 |    85047 |

... 远月合约日行情

    | preclose_minor | open_minor | high_minor | low_minor | close_minor | vol_minor  | amount_minor | oi_minor |
    |----------------|------------|------------|-----------|-------------|------------|--------------|----------|
    |      5393.0000 |  5409.3999 |  5478.7998 | 5388.0000 |   5475.7998 | 24081.0000 | 2613215.2320 |   111382 |
    |      5475.7998 |  5465.0000 |  5469.3999 | 5406.0000 |   5414.7998 | 20018.0000 | 2179312.6400 |   109490 |
    |      5414.7998 |  5412.2002 |  5445.0000 | 5395.2002 |   5429.0000 | 19403.0000 | 2103363.5840 |   112085 |
    |      5429.0000 |  5450.0000 |  5454.7998 | 5390.3999 |   5401.0000 | 18555.0000 | 2010610.0736 |   109955 |
    |      5401.0000 |  5391.3999 |  5404.3999 | 5285.0000 |   5306.0000 | 26264.0000 | 2798834.8928 |   113429 |

... 该品种所有合约的成交量，成交额和持仓量
    | vol_instru | amount_instru | oi_instru |
    |------------|---------------|-----------|
    | 76831.0000 |  8382434.5088 |    251673 |
    | 72764.0000 |  7965087.3344 |    252063 |
    | 70593.0000 |  7696465.1008 |    251269 |
    | 70748.0000 |  7708100.1984 |    252259 |
    | 96976.0000 | 10392377.7536 |    260342 |

... 基差，仓单与现货数据，目前仅股指期货包含现货数据

    |  basis  | basis_rate | basis_rate_annual | in_stock_total | in_stock | available_in_stock |   spot    | spot_diff | spot_diff_rate |
    |---------|------------|-------------------|----------------|----------|--------------------|-----------|-----------|----------------|
    | -4.5530 |    -0.0824 |           -0.8952 |            nan |      nan |                nan | 5528.4470 |   -4.5530 |        -0.0824 |
    |  6.0248 |     0.1100 |            1.2497 |            nan |      nan |                nan | 5478.2248 |    6.0246 |         0.1100 |
    |  7.4140 |     0.1350 |            1.6067 |            nan |      nan |                nan | 5493.4140 |    7.4140 |         0.1350 |
    |  8.0286 |     0.1469 |            1.8361 |            nan |      nan |                nan | 5465.8286 |    8.0288 |         0.1469 |
    | 17.5656 |     0.3262 |            4.2924 |            nan |      nan |                nan | 5384.5656 |   17.5656 |         0.3262 |

"""

import argparse
import numpy as np


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="To translate minor contract, basis, register stock data to '.ss' file"
    )
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--nomp", default=False, action="store_true", help="not using multiprocess, for debug")
    arg_parser.add_argument("--verbose", default=False, action="store_true", help="to print more details")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from config import cfg_path, cfg_strategy
    from hUtils.calendar import CCalendar
    from hUtils.instruments import CInstrumentInfoTable
    from hSolutions.sPreprocess import main_preprocess

    basic_inputs = ["preclose", "open", "high", "low", "close", "vol", "amount", "oi"]
    args = parse_args()
    calendar = CCalendar(cfg_path.path_calendar)
    instru_info_tab = CInstrumentInfoTable(cfg_path.path_instru_info)
    VOL_ALPHA = 0.9

    main_preprocess(
        universe=list(cfg_strategy.universe),
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        basic_inputs=basic_inputs,
        vol_alpha=VOL_ALPHA,
        path_spot_data=cfg_path.path_spot_data,
        path_tsdb_fut=cfg_path.path_tsdb_fut,
        raw_data_by_date_dir=cfg_path.path_basis_and_fundamental,
        preprocess_dir=cfg_path.preprocess_dir,
        calendar=calendar,
        instru_info_tab=instru_info_tab,
        call_multiprocess=not args.nomp,
        verbose=args.verbose,
    )
