import os
import numpy as np
import pandas as pd
from hUtils.tools import qtimer
from hUtils.calendar import CCalendar
from hUtils.ioPlus import PySharedStackPlus

"""
Part I: Macro data: cpi, m2, ppi
"""


def load_macro_data(path_macro_data: str) -> pd.DataFrame:
    return pd.read_excel(path_macro_data, sheet_name="china_cpi_m2")


def reformat_macro(macro_data: pd.DataFrame, hist_bgn_month: int, calendar: CCalendar) -> pd.DataFrame:
    macro_data["trade_month"] = macro_data["trade_month"].map(lambda z: np.int32(z.strftime("%Y%m")))
    macro_data["available_month"] = macro_data["trade_month"].map(lambda z: calendar.get_next_month(z, 2))
    macro_data.set_index("trade_month", inplace=True)
    macro_data = macro_data.truncate(before=hist_bgn_month)
    return macro_data


def merge_macro(reformat_data: pd.DataFrame, dates_header: pd.DataFrame):
    dates_header["available_month"] = dates_header["trading_day"].map(lambda z: z // 100)
    res = pd.merge(
        left=dates_header,
        right=reformat_data,
        on="available_month",
        how="left",
    )
    res.drop(labels="available_month", axis=1, inplace=True)
    return res[["tp", "trading_day", "cpi_rate", "m2_rate", "ppi_rate"]]


@qtimer
def main_macro(
    bgn_date: np.int32,
    end_date: np.int32,
    path_macro_data: str,
    alternative_dir: str,
    calendar: CCalendar,
):
    macro_data_type = np.dtype(
        [
            ("tp", np.int64),
            ("trading_day", np.int32),
            ("cpi_rate", np.float32),
            ("m2_rate", np.float32),
            ("ppi_rate", np.float32),
        ]
    )
    macro_file = "macro.ss"
    macro_path = os.path.join(alternative_dir, macro_file)
    ss = PySharedStackPlus(macro_path, dtype=macro_data_type, push_enable=True)
    if ss.check_daily_continuous(coming_next_date=bgn_date, calendar=calendar) == 0:
        macro_data = load_macro_data(path_macro_data=path_macro_data)
        rft_data = reformat_macro(macro_data=macro_data, hist_bgn_month=201111, calendar=calendar)
        dates_header = calendar.get_dates_header(bgn_date, end_date, timestamp="16:00:00")
        new_macro_data = merge_macro(reformat_data=rft_data, dates_header=dates_header)
        ss.append_from_DataFrame(new_data=new_macro_data, new_data_type=macro_data_type)
        print(new_macro_data)
    return 0


"""
Part II: forex: exchange rate
"""


def load_forex_data(path_forex_data: str) -> pd.DataFrame:
    return pd.read_excel(path_forex_data, sheet_name="USDCNY.CFETS")


def reformat_forex(forex_data: pd.DataFrame) -> pd.DataFrame:
    forex_data["trading_day"] = forex_data["Date"].map(lambda z: np.int32(z.strftime("%Y%m%d")))
    return forex_data


def merge_forex(reformat_data: pd.DataFrame, dates_header: pd.DataFrame):
    res = pd.merge(
        left=dates_header,
        right=reformat_data,
        on="trading_day",
        how="left",
    )
    res.drop(labels="Date", axis=1, inplace=True)
    # "pre_close" in res is different from "preclose" in output *ss file
    #  but it's OK, because only order of columns matters
    #  column names are not used when inserting to .ss file
    #  so no rename methods are necessary
    return res[["tp", "trading_day", "pre_close", "open", "high", "low", "close", "pct_chg"]]


@qtimer
def main_forex(
    bgn_date: np.int32,
    end_date: np.int32,
    path_forex_data: str,
    alternative_dir: str,
    calendar: CCalendar,
):
    forex_data_type = np.dtype(
        [
            ("tp", np.int64),
            ("trading_day", np.int32),
            ("preclose", np.float64),
            ("open", np.float64),
            ("high", np.float64),
            ("low", np.float64),
            ("close", np.float64),
            ("pct_chg", np.float64),
        ]
    )
    forex_file = "forex.ss"
    forex_path = os.path.join(alternative_dir, forex_file)
    ss = PySharedStackPlus(forex_path, dtype=forex_data_type, push_enable=True)
    if ss.check_daily_continuous(coming_next_date=bgn_date, calendar=calendar) == 0:
        forex_data = load_forex_data(path_forex_data=path_forex_data)
        rft_data = reformat_forex(forex_data=forex_data)
        dates_header = calendar.get_dates_header(bgn_date, end_date, timestamp="16:00:00")
        new_forex_data = merge_forex(reformat_data=rft_data, dates_header=dates_header)
        ss.append_from_DataFrame(new_data=new_forex_data, new_data_type=forex_data_type)
        print(new_forex_data)
    return 0
