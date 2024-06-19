"""
0. 会员持仓数据
1. 保存路径为 /var/data/StackData/futures/team/huxo/mbr_pos/cu.ss

|          tp           | trading_day | ticker | info_type | info_rank |    member    | info_qty | info_qty_dlt |
|-----------------------|-------------|--------|-----------|-----------|--------------|----------|--------------|
| 20240322 16:00:00.000 |    20240322 | cu2408 |         3 |        16 |     铜冠金源 |      463 |           -1 |
| 20240322 16:00:00.000 |    20240322 | cu2408 |         3 |        17 |     南华期货 |      452 |           14 |
| 20240322 16:00:00.000 |    20240322 | cu2408 |         3 |        18 |     建信期货 |      434 |            7 |
| 20240322 16:00:00.000 |    20240322 | cu2408 |         3 |        19 |     中粮期货 |      429 |           -1 |
| 20240322 16:00:00.000 |    20240322 | cu2408 |         3 |        20 |     浙商期货 |      349 |           -3 |

"""

import argparse
import numpy as np


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To translate member position data from WDS to .ss file")
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--nomp", default=False, action="store_true", help="not using multiprocess, for debug")
    arg_parser.add_argument("--batch", type=int, default=120, help="number of days processed in a batch")
    arg_parser.add_argument("--verbose", default=False, action="store_true", help="to print more details")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from config import cfg_path, cfg_strategy
    from hUtils.calendar import CCalendar
    from hUtils.instruments import CInstrumentInfoTable
    from hUtils.wds import CDownloadEngineWDS
    from hSolutions.sMbrPos import main_mbr_pos

    args = parse_args()
    PROCESSES = 24
    download_values = [
        "FS_INFO_TYPE",  # type of position, 1:quantity of volume, 2: quantity of buy order 3: quantity of sell order
        "FS_INFO_RANK",  # rank of position
        "FS_INFO_MEMBERNAME",  # member name
        "FS_INFO_POSITIONSNUM",  # quantity
        "S_OI_POSITIONSNUMC",  # delta quantity compared to last trading day
        "S_INFO_WINDCODE",  # wind code, like "MA405.CZC"
    ]

    wds = CDownloadEngineWDS(**cfg_strategy.account_wds)
    calendar = CCalendar(cfg_path.path_calendar)
    instru_info_tab = CInstrumentInfoTable(cfg_path.path_instru_info)
    main_mbr_pos(
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        download_values=download_values,
        wds=wds,
        calendar=calendar,
        instru_info_tab=instru_info_tab,
        mbr_pos_dir=cfg_path.mbr_pos_dir,
        batch_size=args.batch,
        call_multiprocess=not args.nomp,
        processes=PROCESSES,
        verbose=args.verbose,
    )
