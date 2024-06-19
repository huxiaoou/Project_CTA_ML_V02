import os
import datetime as dt
import numpy as np
import pandas as pd
import multiprocessing as mp
from rich.progress import track, Progress
from hUtils.tools import SFG, SFY, qtimer, error_handler
from hUtils.calendar import CCalendar
from hUtils.instruments import CInstrumentInfoTable, parse_instrument_from_contract
from hUtils.ioPlus import load_tsdb
from hUtils.ioPlus import PySharedStackPlus

"""
Part I: load fundamental data: basis and stock
"""


def load_basis_by_date(
    basis_file_tmpl: str, trading_day: np.int32, raw_data_by_date_dir: str, basis_cols: list[str]
) -> pd.DataFrame:
    basis_file = basis_file_tmpl.format(trading_day)
    basis_path = os.path.join(raw_data_by_date_dir, str(trading_day)[0:4], str(trading_day), basis_file)
    basis_data = pd.read_csv(basis_path)
    basis_data["tp"] = dt.datetime.strptime(f"{trading_day} 16:00:00", "%Y%m%d %H:%M:%S").timestamp() * 10**9
    basis_data["trading_day"] = trading_day
    return basis_data[["tp", "trading_day", "instrument"] + basis_cols]


def load_basis(
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    raw_data_by_date_dir: str,
    basis_cols: list[str],
) -> pd.DataFrame:
    cfe_cols = basis_cols
    oth_cols = basis_cols[:-1]

    cfe_dfs: list[pd.DataFrame] = []
    oth_dfs: list[pd.DataFrame] = []
    iter_dates = calendar.get_iter_list(bgn_date, end_date)
    for trading_day in track(iter_dates, description=f"[INF] Loading basis {bgn_date}->{end_date}"):
        cfe_basis_data = load_basis_by_date("basis_cfe.{}.csv.gz", trading_day, raw_data_by_date_dir, cfe_cols)
        cfe_dfs.append(cfe_basis_data)
        oth_basis_data = load_basis_by_date("basis.{}.csv.gz", trading_day, raw_data_by_date_dir, oth_cols)
        oth_dfs.append(oth_basis_data)

    cfe_df = pd.concat(cfe_dfs, axis=0, ignore_index=True)
    oth_df = pd.concat(oth_dfs, axis=0, ignore_index=True)

    # cfe adjustment
    cfe_df["instrument"] = cfe_df["instrument"].str.upper()
    cfe_df[cfe_cols] = -cfe_df[cfe_cols].fillna(0)

    # concat
    df = pd.concat([cfe_df, oth_df], axis=0, ignore_index=True)
    df.sort_values(by=["trading_day", "instrument"], ascending=True, inplace=True)
    return df


def load_stock(
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    raw_data_by_date_dir: str,
    stock_cols: list[str],
) -> pd.DataFrame:

    # load by date
    rs_dfs: list[pd.DataFrame] = []
    iter_dates = calendar.get_iter_list(bgn_date, end_date)
    for trading_day in track(iter_dates, description=f"[INF] Loading stock {bgn_date}->{end_date}"):
        rs_file = f"stock.{trading_day}.csv.gz"
        rs_path = os.path.join(raw_data_by_date_dir, str(trading_day)[0:4], str(trading_day), rs_file)
        rs_data = pd.read_csv(rs_path)
        rs_data["tp"] = dt.datetime.strptime(f"{trading_day} 16:00:00", "%Y%m%d %H:%M:%S").timestamp() * 10**9
        rs_data["trading_day"] = trading_day
        rs_data = rs_data[["tp", "trading_day", "instrument"] + stock_cols]
        rs_dfs.append(rs_data)

    # concat
    df = pd.concat(rs_dfs, axis=0, ignore_index=True)
    df.sort_values(by=["trading_day", "instrument"], ascending=True, inplace=True)
    return df


def load_spot(path_spot_data: str) -> pd.DataFrame:
    spot_data = pd.read_excel(path_spot_data, header=2)
    spot_data["trading_day"] = spot_data["trading_day"].map(lambda z: np.int32(z.strftime("%Y%m%d")))
    spot_data["tp"] = spot_data["trading_day"].map(
        lambda z: dt.datetime.strptime(f"{z} 16:00:00", "%Y%m%d %H:%M:%S").timestamp() * 10**9
    )
    return spot_data


"""
Part II: load major data
"""


@qtimer
def reformat(raw_data: pd.DataFrame, instru_info_tab: CInstrumentInfoTable, basic_inputs: list[str]) -> pd.DataFrame:
    raw_data["instrument"] = raw_data["ticker"].map(lambda z: parse_instrument_from_contract(z.decode("utf-8")))
    raw_data["ticker"] = raw_data[["ticker", "trading_day"]].apply(
        lambda z: instru_info_tab.fix_contract_id(
            ticker=z["ticker"].decode("utf-8"), trade_date=str(z["trading_day"])
        ).encode("utf-8"),
        axis=1,
    )
    return raw_data[["tp", "trading_day", "ticker"] + basic_inputs + ["instrument"]]


def to_dict_by_instru(
    reformat_data: pd.DataFrame,
    universe: list[str],
    default_val: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    res = {}
    for instru, instru_df in reformat_data.groupby(by="instrument"):
        res[instru] = instru_df.drop(labels="instrument", axis=1)
    for instru in universe:
        if instru not in res:
            res[instru] = default_val
    return res


def spot_to_dict_by_instru(
    spot_data: pd.DataFrame,
    universe: list[str],
    default_val: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    res = {}
    for instru in spot_data.columns:
        if instru not in ["tp", "trading_day"]:
            res[instru] = spot_data[["tp", "trading_day", instru]].rename(mapper={instru: "spot"}, axis=1)
    for instru in universe:
        if instru not in res:
            res[instru] = default_val
    return res


"""
Part III: found major and minor ticker
"""


def find_major_and_minor_by_instru(
    instru: str,
    instru_all_data: pd.DataFrame,
    vol_alpha: float,
    basic_inputs: list[str],
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    return: 2 pd.DataFrames with cols = minor_cols =  ["tp", "trading_day", "ticker"] + basic_inputs
            first for major, second for minor

    """

    def __reformat(raw_data: pd.DataFrame, cols: list[str]):
        if raw_data.empty:
            return pd.DataFrame(columns=cols)
        else:
            raw_data = raw_data.reset_index().rename(mapper={"index": "ticker"}, axis=1)
            return raw_data[cols]

    major_res, minor_res = [], []
    if not instru_all_data.empty:
        instru_all_data["oi_add_vol"] = (
            instru_all_data["oi"].fillna(0) * (1 - vol_alpha) + instru_all_data["vol"].fillna(0) * vol_alpha
        )
        instru_all_data = instru_all_data.sort_values(
            by=["trading_day", "oi_add_vol", "ticker"],
            ascending=[True, False, True],
        ).set_index("ticker")
        for (tp, trading_day), trading_day_instru_data in instru_all_data.groupby(by=["tp", "trading_day"]):
            sv = trading_day_instru_data["oi_add_vol"]  #  a pd.Series: sum of oi and vol, with contract_id as index
            major_ticker = sv.idxmax()
            minor_sv = sv[sv.index > major_ticker]
            if not minor_sv.empty:
                minor_ticker = minor_sv.idxmax()
            else:
                minor_sv = sv[sv.index < major_ticker]
                if not minor_sv.empty:
                    minor_ticker = minor_sv.idxmax()
                    # always keep major_ticker is ahead of minor_ticker
                    major_ticker, minor_ticker = minor_ticker, major_ticker
                else:
                    minor_ticker = major_ticker
                    if verbose:
                        print(f"[INF] There is only one ticker for {SFY(instru)} at {SFG(str(trading_day))}")
            s_major = trading_day_instru_data.loc[major_ticker]
            s_minor = trading_day_instru_data.loc[minor_ticker]
            major_res.append(s_major)
            minor_res.append(s_minor)
    major_data, minor_data = pd.DataFrame(major_res), pd.DataFrame(minor_res)
    reformat_cols = ["tp", "trading_day", "ticker"] + basic_inputs
    major_data, minor_data = __reformat(major_data, reformat_cols), __reformat(minor_data, reformat_cols)
    return major_data, minor_data


"""
Part IV: found vol, amount, oi by instrument
"""


def sum_vol_amount_oi_by_instru(instru_all_data: pd.DataFrame) -> pd.DataFrame:
    vol_amount_oi_cols = ["vol", "amount", "oi"]
    save_cols = ["tp", "trading_day"] + vol_amount_oi_cols
    if not instru_all_data.empty:
        sum_df = pd.pivot_table(
            data=instru_all_data,
            index=["tp", "trading_day"],
            values=vol_amount_oi_cols,
            aggfunc="sum",
        )
        sum_df = sum_df.reset_index()
        return sum_df[save_cols]
    else:
        return pd.DataFrame(columns=save_cols)


"""
Part V: Misc vol, amount, and oi, adjustment
"""


def make_double_to_single(
    instru_data: pd.DataFrame,
    adj_cols: list[str],
    adj_date: int,
    instru: str,
    exempt_instruments: list[str],
):
    if instru in exempt_instruments:
        adj_ratio = 1
    else:
        adj_ratio = [2 if t < adj_date else 1 for t in instru_data["trading_day"]]
    instru_data[adj_cols] = instru_data[adj_cols].div(adj_ratio, axis="index").fillna(0)
    # fillna(0) is necessary
    # for oi, oi_minor, oi_instru: np cannot convert float NaN to integer, 0 must be provided
    return 0


def merge_all(
    dates_header: pd.DataFrame,
    instru_maj_data: pd.DataFrame,
    instru_min_data: pd.DataFrame,
    instru_vol_data: pd.DataFrame,
    instru_basis_data: pd.DataFrame,
    instru_stock_data: pd.DataFrame,
    instru_spot_data: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["tp", "trading_day"]
    merged_data = pd.merge(left=dates_header, right=instru_maj_data, on=keys, how="left")
    merged_data = merged_data.merge(right=instru_min_data, on=keys, how="left", suffixes=("_major", "_minor"))
    merged_data = merged_data.merge(right=instru_vol_data, on=keys, how="left")
    merged_data = merged_data.merge(right=instru_basis_data, on=keys, how="left")
    merged_data = merged_data.merge(right=instru_stock_data, on=keys, how="left")
    merged_data = merged_data.merge(right=instru_spot_data, on=keys, how="left")
    return merged_data


def adjust_and_select(
    instru: str,
    merged_data: pd.DataFrame,
    basis_cols: list[str],
    stock_cols: list[str],
    spot_cols: list[str],
    basic_inputs: list[str],
) -> pd.DataFrame:
    __vol_adj_date: int = 20200101
    __amt_shrink_scale: float = 1e4
    _spot_diff_rate_scale: float = 100

    # adjust volume, amount and oi cols
    __vol_cols = ["vol", "amount", "oi"]
    merged_data.rename(mapper={z: f"{z}_instru" for z in __vol_cols}, axis=1, inplace=True)

    adj_cols = (
        [f"{z}_major" for z in __vol_cols] + [f"{z}_minor" for z in __vol_cols] + [f"{z}_instru" for z in __vol_cols]
    )
    make_double_to_single(
        instru_data=merged_data,
        adj_cols=adj_cols,
        adj_date=__vol_adj_date,
        instru=instru,
        exempt_instruments=["IH", "IF", "IC", "IM", "TF", "TS", "T", "TL"],
    )

    # shrink amount from YUAN to WANYUAN
    amt_cols = ["amount_major", "amount_minor", "amount_instru"]
    merged_data[amt_cols] = merged_data[amt_cols].astype(np.float64) / __amt_shrink_scale

    # spot
    merged_data["spot_diff"] = merged_data["spot"] - merged_data["close_major"]
    merged_data["spot_diff_rate"] = merged_data["spot_diff"] / merged_data["spot"] * _spot_diff_rate_scale

    # fill nan ticker
    merged_data[["ticker_major", "ticker_minor"]] = merged_data[["ticker_major", "ticker_minor"]].fillna(value=b"")

    # selected data
    header_cols = ["tp", "trading_day", "ticker_major", "ticker_minor"]
    major_inputs = [z + "_major" for z in basic_inputs]
    minor_inputs = [z + "_minor" for z in basic_inputs]
    instru_sum_cols = ["vol_instru", "amount_instru", "oi_instru"]
    selected_cols = header_cols + major_inputs + minor_inputs + instru_sum_cols + basis_cols + stock_cols + spot_cols
    selected_data = merged_data[selected_cols]
    return selected_data


def process_for_instru(
    instru: str,
    bgn_date: np.int32,
    basic_inputs: list[str],
    vol_alpha: float,
    instru_basis_data: pd.DataFrame,
    instru_stock_data: pd.DataFrame,
    instru_all_data: pd.DataFrame,
    instru_spot_data: pd.DataFrame,
    dates_header: pd.DataFrame,
    preprocess_dir: str,
    new_data_type: np.dtype,
    basis_cols: list[str],
    stock_cols: list[str],
    spot_cols: list[str],
    calendar: CCalendar,
    verbose: bool,
):
    # for instru in universe:
    preprocess_file = f"{instru}.ss"
    preprocess_path = os.path.join(preprocess_dir, preprocess_file)
    ss = PySharedStackPlus(preprocess_path, dtype=new_data_type, push_enable=True)
    if ss.check_daily_continuous(coming_next_date=bgn_date, calendar=calendar) == 0:
        instru_maj_data, instru_min_data = find_major_and_minor_by_instru(
            instru=instru,
            instru_all_data=instru_all_data,
            basic_inputs=basic_inputs,
            vol_alpha=vol_alpha,
            verbose=verbose,
        )
        instru_vol_data = sum_vol_amount_oi_by_instru(instru_all_data=instru_all_data)
        merged_data = merge_all(
            dates_header=dates_header,
            instru_maj_data=instru_maj_data,
            instru_min_data=instru_min_data,
            instru_vol_data=instru_vol_data,
            instru_basis_data=instru_basis_data,
            instru_stock_data=instru_stock_data,
            instru_spot_data=instru_spot_data,
        )
        new_data = adjust_and_select(
            instru=instru,
            merged_data=merged_data,
            basis_cols=basis_cols,
            stock_cols=stock_cols,
            spot_cols=spot_cols,
            basic_inputs=basic_inputs,
        )
        ss.append_from_DataFrame(new_data=new_data, new_data_type=new_data_type)
    return 0


@qtimer
def main_preprocess(
    universe: list[str],
    bgn_date: np.int32,
    end_date: np.int32,
    basic_inputs: list[str],
    vol_alpha: float,
    path_spot_data: str,
    path_tsdb_fut: str,
    raw_data_by_date_dir: str,
    preprocess_dir: str,
    calendar: CCalendar,
    instru_info_tab: CInstrumentInfoTable,
    call_multiprocess: bool,
    verbose: bool,
):
    # dtype settings
    __basic_input_dtype = {
        "preclose": np.float64,
        "open": np.float64,
        "high": np.float64,
        "low": np.float64,
        "close": np.float64,
        "vol": np.float64,
        "amount": np.float64,
        "oi": np.int32,
    }
    vol_amount_oi_cols = ["vol", "amount", "oi"]
    basis_cols = ["basis", "basis_rate", "basis_rate_annual"]
    stock_cols = ["in_stock_total", "in_stock", "available_in_stock"]
    spot_cols = ["spot", "spot_diff", "spot_diff_rate"]
    new_data_type = np.dtype(
        [
            ("tp", np.int64),
            ("trading_day", np.int32),
            ("ticker_major", "S8"),
            ("ticker_minor", "S8"),
        ]
        + [(v + "_major", __basic_input_dtype[v]) for v in basic_inputs]
        + [(v + "_minor", __basic_input_dtype[v]) for v in basic_inputs]
        + [(v + "_instru", __basic_input_dtype[v]) for v in vol_amount_oi_cols]
        + [(v, np.float64) for v in basis_cols]
        + [(v, np.float32) for v in stock_cols]
        + [(v, np.float64) for v in spot_cols]
    )

    # load all data
    raw_all_data = load_tsdb(path_tsdb_fut, "d01e", basic_inputs, bgn_date, end_date)
    rft_all_data = reformat(raw_data=raw_all_data, instru_info_tab=instru_info_tab, basic_inputs=basic_inputs)
    all_data_by_instru = to_dict_by_instru(
        reformat_data=rft_all_data,
        universe=universe,
        default_val=pd.DataFrame(columns=["tp", "trading_day", "ticker"] + basic_inputs),
    )

    # load basis
    basis_rft = load_basis(bgn_date, end_date, calendar, raw_data_by_date_dir, basis_cols=basis_cols)
    basis_by_instru = to_dict_by_instru(
        reformat_data=basis_rft,
        universe=universe,
        default_val=pd.DataFrame(columns=["tp", "trading_day"] + basis_cols),
    )

    # load register stock
    stock_rft = load_stock(bgn_date, end_date, calendar, raw_data_by_date_dir, stock_cols=stock_cols)
    stock_by_instru = to_dict_by_instru(
        reformat_data=stock_rft,
        universe=universe,
        default_val=pd.DataFrame(columns=["tp", "trading_day"] + stock_cols),
    )

    # load spot
    spot_data = load_spot(path_spot_data)
    spot_data_by_instru = spot_to_dict_by_instru(
        spot_data=spot_data,
        universe=universe,
        default_val=pd.DataFrame(columns=["tp", "trading_day", "spot"]),
    )

    # header
    dates_header = calendar.get_dates_header(bgn_date, end_date, timestamp="16:00:00")

    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=f"[INF] Preprocessing {bgn_date}->{end_date}", total=len(universe))
            with mp.get_context("spawn").Pool() as pool:
                for instru in universe:
                    instru_basis_data = basis_by_instru[instru]
                    instru_stock_data = stock_by_instru[instru]
                    instru_all_data = all_data_by_instru[instru]
                    instru_spot_data = spot_data_by_instru[instru]
                    pool.apply_async(
                        process_for_instru,
                        kwds={
                            "instru": instru,
                            "bgn_date": bgn_date,
                            "basic_inputs": basic_inputs,
                            "vol_alpha": vol_alpha,
                            "instru_basis_data": instru_basis_data,
                            "instru_stock_data": instru_stock_data,
                            "instru_all_data": instru_all_data,
                            "instru_spot_data": instru_spot_data,
                            "dates_header": dates_header,
                            "preprocess_dir": preprocess_dir,
                            "new_data_type": new_data_type,
                            "basis_cols": basis_cols,
                            "stock_cols": stock_cols,
                            "spot_cols": spot_cols,
                            "calendar": calendar,
                            "verbose": verbose,
                        },
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for instru in track(universe, description=f"Preprocessing {bgn_date}->{end_date}"):
            # for instru in universe:
            instru_basis_data = basis_by_instru[instru]
            instru_stock_data = stock_by_instru[instru]
            instru_all_data = all_data_by_instru[instru]
            instru_spot_data = spot_data_by_instru[instru]
            process_for_instru(
                instru=instru,
                bgn_date=bgn_date,
                basic_inputs=basic_inputs,
                vol_alpha=vol_alpha,
                instru_basis_data=instru_basis_data,
                instru_stock_data=instru_stock_data,
                instru_all_data=instru_all_data,
                instru_spot_data=instru_spot_data,
                dates_header=dates_header,
                preprocess_dir=preprocess_dir,
                new_data_type=new_data_type,
                basis_cols=basis_cols,
                stock_cols=stock_cols,
                spot_cols=spot_cols,
                calendar=calendar,
                verbose=verbose,
            )
    return 0
