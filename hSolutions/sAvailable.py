import os
import datetime as dt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from hUtils.tools import qtimer
from hUtils.ioPlus import PySharedStackPlus
from hUtils.calendar import CCalendar


@dataclass(frozen=True)
class CCfgAvlbUnvrs:
    universe: dict[str, dict[str, str]]
    win: int
    amount_threshold: float


def load_major(instru: str, major_dir: str) -> pd.DataFrame:
    major_file = f"{instru}.ss"
    major_path = os.path.join(major_dir, major_file)
    ss = PySharedStackPlus(major_path)
    return pd.DataFrame(ss.read_all())


def get_available_universe_by_date(x: pd.Series, ret_df: pd.DataFrame, amt_df: pd.DataFrame) -> pd.DataFrame:
    # x is a Series like this: pd.Series({"cu":True, "CY":False}, name=20120104)
    trading_day: np.int32 = x.name  # type:ignore
    sub_available_universe_df = pd.DataFrame(
        {
            "return": ret_df.loc[trading_day, x],  # type:ignore
            "amount": amt_df.loc[trading_day, x],  # type:ignore
        }
    )
    sub_available_universe_df["trading_day"] = trading_day
    return sub_available_universe_df


def get_available_universe(
    bgn_date: np.int32,
    end_date: np.int32,
    major_dir: str,
    cfg_avlb_unvrs: CCfgAvlbUnvrs,
    calendar: CCalendar,
) -> pd.DataFrame:
    win_start_date = calendar.get_next_date(bgn_date, -cfg_avlb_unvrs.win + 1)
    amt_data, amt_ma_data, return_data = {}, {}, {}
    for instru in cfg_avlb_unvrs.universe:
        instru_major_data = load_major(instru=instru, major_dir=major_dir)
        selected_major_data = instru_major_data.set_index("trading_day").truncate(
            before=int(win_start_date), after=int(end_date)
        )
        amt_ma_data[instru] = selected_major_data["amount"].fillna(0).rolling(window=cfg_avlb_unvrs.win).mean()
        amt_data[instru] = selected_major_data["amount"].fillna(0)
        return_data[instru] = selected_major_data["major_return"]

    # --- reorganize and save
    amt_df, amt_ma_df, return_df = pd.DataFrame(amt_data), pd.DataFrame(amt_ma_data), pd.DataFrame(return_data)
    filter_df: pd.DataFrame = amt_ma_df >= cfg_avlb_unvrs.amount_threshold
    filter_df = filter_df.truncate(before=int(bgn_date))
    res = filter_df.apply(get_available_universe_by_date, args=(return_df, amt_df), axis=1)  # type:ignore
    update_df: pd.DataFrame = pd.concat(res.tolist(), axis=0, ignore_index=False).reset_index()
    update_df.rename(mapper={"index": "instrument"}, axis=1, inplace=True)
    update_df["tp"] = update_df["trading_day"].map(
        lambda z: dt.datetime.strptime(f"{z} 16:00:00", "%Y%m%d %H:%M:%S").timestamp() * 10**9
    )
    update_df.sort_values(by=["tp", "trading_day", "amount"], ascending=[True, True, False], inplace=True)
    update_df = update_df[["tp", "trading_day", "instrument", "return", "amount"]]

    # --- add section
    unvrs = cfg_avlb_unvrs.universe
    update_df["sectorL0"] = update_df["instrument"].map(lambda z: unvrs[z]["sectorL0"].encode("utf-8"))
    update_df["sectorL1"] = update_df["instrument"].map(lambda z: unvrs[z]["sectorL1"].encode("utf-8"))

    # --- encode instrument
    update_df["instrument"] = update_df["instrument"].str.encode("utf-8")
    return update_df


@qtimer
def main_available(
    bgn_date: np.int32,
    end_date: np.int32,
    cfg_avlb_unvrs: CCfgAvlbUnvrs,
    available_dir: str,
    major_dir: str,
    calendar: CCalendar,
):
    ss_dtype = np.dtype(
        [
            ("tp", np.int64),
            ("trading_day", np.int32),
            ("instrument", "S4"),
            ("return", np.float64),
            ("amount", np.float32),
            ("sectorL0", "S4"),
            ("sectorL1", "S4"),
        ]
    )
    ss_path = os.path.join(available_dir, "available.ss")
    ss = PySharedStackPlus(ss_path, dtype=ss_dtype, push_enable=True)
    if ss.check_daily_continuous(bgn_date, calendar) == 0:
        new_data = get_available_universe(
            bgn_date=bgn_date,
            end_date=end_date,
            major_dir=major_dir,
            cfg_avlb_unvrs=cfg_avlb_unvrs,
            calendar=calendar,
        )
        print(new_data)
        ss.append_from_DataFrame(new_data=new_data, new_data_type=ss_dtype)
    return 0
