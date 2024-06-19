import os
import datetime as dt
import numpy as np
import pandas as pd
from hUtils.tools import qtimer
from hUtils.ioPlus import PySharedStackPlus
from hUtils.calendar import CCalendar


def load_available(available_dir: str) -> pd.DataFrame:
    available_file = "available.ss"
    available_path = os.path.join(available_dir, available_file)
    ss = PySharedStackPlus(available_path)
    return pd.DataFrame(ss.read_all())


def cal_market_return_by_date(sub_data: pd.DataFrame) -> float:
    wgt = sub_data["rel_wgt"] / sub_data["rel_wgt"].sum()
    ret = sub_data["return"] @ wgt
    return ret


def cal_market_return(bgn_date: np.int32, end_date: np.int32, available_dir: str) -> pd.DataFrame:
    available_data = load_available(available_dir=available_dir)
    input_for_return = available_data.set_index("trading_day").truncate(before=int(bgn_date), after=int(end_date))
    input_for_return["rel_wgt"] = np.sqrt(input_for_return["amount"])
    ret = {}
    ret["market"] = input_for_return.groupby(by="trading_day").apply(cal_market_return_by_date)
    for sector_byte, sector_df in input_for_return.groupby(by="sectorL0"):
        sector = sector_byte.decode("utf-8")  # type: ignore
        ret[sector] = sector_df.groupby(by="trading_day").apply(cal_market_return_by_date)
    for sector_byte, sector_df in input_for_return.groupby(by="sectorL1"):
        sector = sector_byte.decode("utf-8")  # type: ignore
        ret[sector] = sector_df.groupby(by="trading_day").apply(cal_market_return_by_date)
    ret_by_sector = pd.DataFrame(ret)

    # reformat
    ret_by_sector = ret_by_sector.reset_index()
    mkt_cols = ["market"]
    sec0_cols = ["C", "F"]
    sec1_cols = ["GLD", "MTL", "OIL", "CHM", "BLK", "AGR", "EQT"]
    ret_by_sector = ret_by_sector[["trading_day"] + mkt_cols + sec0_cols + sec1_cols]
    return ret_by_sector


def load_market_index(
    bgn_date: np.int32,
    end_date: np.int32,
    path_mkt_idx_data: str,
    mkt_idxes: list[str],
) -> pd.DataFrame:
    mkt_idx_data = {}
    for mkt_idx in mkt_idxes:
        df = pd.read_excel(path_mkt_idx_data, sheet_name=mkt_idx, header=1)
        df["trading_day"] = df["Date"].map(lambda _: np.int32(_.strftime("%Y%m%d")))
        mkt_idx_data[mkt_idx] = df.set_index("trading_day")["pct_chg"] / 100
    mkt_idx_df = pd.DataFrame(mkt_idx_data).truncate(before=int(bgn_date), after=int(end_date))
    mkt_idx_df = mkt_idx_df.reset_index()
    return mkt_idx_df


def merge_mkt_idx(ret_by_sector: pd.DataFrame, mkt_idx_df: pd.DataFrame) -> pd.DataFrame:
    if (s0 := len(ret_by_sector)) != (s1 := len(mkt_idx_df)):
        print(f"[INF] length of custom market index = {s0}")
        print(f"[INF] length of        market index = {s1}")
        d0 = set(ret_by_sector["trading_day"])
        d1 = set(mkt_idx_df["trading_day"])
        in_d0_not_in_d1 = d0 - d1
        in_d1_not_in_d0 = d1 - d0
        if in_d0_not_in_d1:
            print(f"[INF] the following days are in custom but not in official {in_d0_not_in_d1}")
        if in_d1_not_in_d0:
            print(f"[INF] the following days are in official but not in custom {in_d1_not_in_d0}")
    new_data = pd.merge(left=ret_by_sector, right=mkt_idx_df, on="trading_day", how="right")
    new_data["tp"] = new_data["trading_day"].map(
        lambda z: dt.datetime.strptime(f"{z} 16:00:00", "%Y%m%d %H:%M:%S").timestamp() * 10**9
    )
    new_data = new_data.set_index("tp").reset_index()
    return new_data


@qtimer
def main_market(
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    available_dir: str,
    market_dir: str,
    path_mkt_idx_data: str,
    mkt_idxes: list[str],
):
    l0 = [
        ("tp", np.int64),
        ("trading_day", np.int32),
        ("market", np.float64),
        ("C", np.float64),
        ("F", np.float64),
        ("GLD", np.float64),
        ("MTL", np.float64),
        ("OIL", np.float64),
        ("CHM", np.float64),
        ("BLK", np.float64),
        ("AGR", np.float64),
        ("EQT", np.float64),
    ]
    l1 = [(m, np.float64) for m in mkt_idxes]
    ss_dtype = np.dtype(l0 + l1)
    ss_path = os.path.join(market_dir, "market.ss")
    ss = PySharedStackPlus(ss_path, dtype=ss_dtype, push_enable=True)
    if ss.check_daily_continuous(bgn_date, calendar) == 0:
        ret_by_sector = cal_market_return(bgn_date, end_date, available_dir)
        mkt_idx_df = load_market_index(bgn_date, end_date, path_mkt_idx_data, mkt_idxes)
        new_data = merge_mkt_idx(ret_by_sector, mkt_idx_df)
        print(new_data)
        ss.append_from_DataFrame(new_data=new_data, new_data_type=ss_dtype)
    return 0
