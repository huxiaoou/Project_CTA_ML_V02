import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from rich.progress import track, Progress
from hUtils.tools import qtimer, SFG, error_handler
from hUtils.calendar import CCalendar
from hUtils.instruments import CInstrumentInfoTable, parse_instrument_from_contract
from hUtils.ioPlus import PySharedStackPlus, load_tsdb
from hUtils.typeDef import TPatchData


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


def get_preprice(instru_md_data: pd.DataFrame, price: str) -> pd.DataFrame:
    """
    params: instru_md_data: a pd.DataFrame with columns = [["tp", "trading_day", "ticker"] + basic_inputs]
    params: price: must be in  basic_inputs, usually would be "open"

    return : a pd.DataFrame with columns = ["tp", "trading_day", "ticker", f"pre{price}"]

    """
    pivot_data = pd.pivot_table(
        data=instru_md_data,
        index=["tp", "trading_day"],
        columns="ticker",
        values=price,
        aggfunc=pd.Series.mean,
    )
    pivot_pre_data = pivot_data.sort_index(ascending=True).shift(1)
    pre_price_data = pivot_pre_data.stack().reset_index().rename(mapper={0: f"pre{price}"}, axis=1)
    return pre_price_data


def merge_all(
    dates_header: pd.DataFrame,
    instru_data: pd.DataFrame,
    preopen_data: pd.DataFrame,
    preprices: list[str],
    basic_inputs: list[str],
) -> pd.DataFrame:
    res = pd.merge(left=dates_header, right=instru_data, on=["tp", "trading_day"], how="left")
    res = res.merge(right=preopen_data, on=["tp", "trading_day", "ticker"], how="left")
    res["ticker"] = res["ticker"].fillna(b"")
    selected_cols = ["tp", "trading_day", "ticker"] + preprices + basic_inputs
    selected_data = res[selected_cols]
    return selected_data


def get_init_close_val(instru_data: pd.DataFrame, ss: PySharedStackPlus) -> np.float64:
    ini_val_from_instru = np.float64(np.nan) if instru_data.empty else instru_data["preclose"].iloc[0]
    if ss.empty:
        return ini_val_from_instru
    else:
        ini_val_from_ss = ss.last_price_not_nan(price="closeM")
        return ini_val_from_ss or ini_val_from_instru


def core(
    instru: str,
    instru_data: pd.DataFrame,
    init_close_val: np.float64,
    basic_inputs: list[str],
    patch_data: TPatchData,
    amount_shrink_scale: float = 1e4,
    vol_adj_date: int = 20200101,
) -> pd.DataFrame:
    """
    params path_data: dict[instru:str, patch_data:dict]
                      patch_data is a dice like this:

                      keys: tuple[np.int32, bytes] like (20220310, b'ni2204')
                      vals: dict[str, float | int] like ()
    """

    if instru_patch_data := patch_data.get(instru):
        idx_cols = ["trading_day", "ticker"]
        patch_df = pd.DataFrame.from_dict(instru_patch_data, orient="index")
        instru_data = instru_data.set_index(idx_cols)
        instru_data.update(patch_df)
        instru_data = instru_data.reset_index()

    # --- adjust volume amount openInterest
    if instru in ["IH", "IF", "IC", "IM", "TF", "TS", "T", "TL"]:
        instru_data["vol_adj_ratio"] = 1
    else:
        instru_data["vol_adj_ratio"] = [2 if t < vol_adj_date else 1 for t in instru_data["trading_day"]]
    adj_cols = ["vol", "amount", "oi"]
    instru_data[adj_cols] = instru_data[adj_cols].div(instru_data["vol_adj_ratio"], axis="index").fillna(0)
    instru_data["amount"] = instru_data["amount"] / amount_shrink_scale

    # --- major return
    instru_data["major_return"] = (
        instru_data["close"].astype(np.float64) / instru_data["preclose"].astype(np.float64) - 1
    )
    instru_data["major_return_open"] = (
        instru_data["open"].astype(np.float64) / instru_data["preopen"].astype(np.float64) - 1
    )

    # --- continuous instrument price index
    instru_data["closeM"] = (instru_data["major_return"] + 1).cumprod() * init_close_val  # type: ignore
    instru_data["openM"] = instru_data["closeM"] * instru_data["open"] / instru_data["close"]
    instru_data["highM"] = instru_data["closeM"] * instru_data["high"] / instru_data["close"]
    instru_data["lowM"] = instru_data["closeM"] * instru_data["low"] / instru_data["close"]

    # --- select columns
    major_cols = ["major_return", "major_return_open", "openM", "highM", "lowM", "closeM"]
    expand_cols = ["tp", "trading_day", "ticker"] + ["preopen"] + basic_inputs + major_cols
    expand_data = instru_data[expand_cols]
    return expand_data


def process_for_instrument(
    instru: str,
    instru_maj_data: pd.DataFrame,
    instru_all_data: pd.DataFrame,
    dates_header: pd.DataFrame,
    basic_inputs: list[str],
    patch_data: TPatchData,
    major_dir: str,
    major_data_type: np.dtype,
    bgn_date: np.int32,
    calendar: CCalendar,
):
    major_file = f"{instru}.ss"
    major_path = os.path.join(major_dir, major_file)
    ss = PySharedStackPlus(major_path, dtype=major_data_type, push_enable=True)
    if ss.check_daily_continuous(coming_next_date=bgn_date, calendar=calendar) == 0:
        preprices_data = get_preprice(instru_md_data=instru_all_data, price="open")
        init_close_val = get_init_close_val(instru_data=instru_maj_data, ss=ss)
        instru_data = merge_all(
            dates_header=dates_header,
            instru_data=instru_maj_data,
            preopen_data=preprices_data,
            preprices=["preopen"],
            basic_inputs=basic_inputs,
        )
        new_major_data = core(instru, instru_data, init_close_val, basic_inputs, patch_data=patch_data)
        ss.append_from_DataFrame(new_data=new_major_data, new_data_type=major_data_type)
    return 0


@qtimer
def main_major(
    universe: list[str],
    bgn_date: np.int32,
    end_date: np.int32,
    basic_inputs: list[str],
    path_tsdb_futhot: str,
    path_tsdb_fut: str,
    major_dir: str,
    calendar: CCalendar,
    instru_info_tab: CInstrumentInfoTable,
    patch_data: TPatchData,
    call_multiprocess: bool,
):
    major_data_type = np.dtype(
        [
            ("tp", np.int64),
            ("trading_day", np.int32),
            ("ticker", "S8"),
            ("preopen", np.float64),
            ("preclose", np.float64),
            ("open", np.float64),
            ("high", np.float64),
            ("low", np.float64),
            ("close", np.float64),
            ("vol", np.float64),
            ("amount", np.float64),
            ("oi", np.int32),
            ("major_return", np.float64),
            ("major_return_open", np.float64),
            ("openM", np.float64),
            ("highM", np.float64),
            ("lowM", np.float64),
            ("closeM", np.float64),
        ]
    )

    # load all hot data
    raw_data = load_tsdb(path_tsdb_futhot, "d01e", basic_inputs, bgn_date, end_date)
    rft_data = reformat(raw_data=raw_data, instru_info_tab=instru_info_tab, basic_inputs=basic_inputs)
    maj_data_by_instru = to_dict_by_instru(
        reformat_data=rft_data,
        universe=universe,
        default_val=pd.DataFrame(columns=["tp", "trading_day", "ticker"] + basic_inputs),
    )

    # load all market data, to append preopen
    base_bgn_date = calendar.get_next_date(bgn_date, -1)
    raw_all_data = load_tsdb(path_tsdb_fut, "d01e", basic_inputs, base_bgn_date, end_date)
    rft_all_data = reformat(raw_data=raw_all_data, instru_info_tab=instru_info_tab, basic_inputs=basic_inputs)
    all_data_by_instru = to_dict_by_instru(
        reformat_data=rft_all_data,
        universe=universe,
        default_val=pd.DataFrame(columns=["tp", "trading_day", "ticker"] + basic_inputs),
    )

    # get dates header
    dates_header = calendar.get_dates_header(bgn_date, end_date, timestamp="16:00:00")

    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description="[INF] Generating major data by instruments", total=len(universe))
            with mp.get_context("spawn").Pool() as pool:
                for instru in universe:
                    instru_maj_data = maj_data_by_instru[instru]
                    instru_all_data = all_data_by_instru[instru]
                    pool.apply_async(
                        process_for_instrument,
                        kwds={
                            "instru": instru,
                            "instru_maj_data": instru_maj_data,
                            "instru_all_data": instru_all_data,
                            "dates_header": dates_header,
                            "basic_inputs": basic_inputs,
                            "patch_data": patch_data,
                            "major_dir": major_dir,
                            "major_data_type": major_data_type,
                            "bgn_date": bgn_date,
                            "calendar": calendar,
                        },
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for instru in track(universe, description="[INF] Generating major data by instruments"):
            instru_maj_data = maj_data_by_instru[instru]
            instru_all_data = all_data_by_instru[instru]
            process_for_instrument(
                instru=instru,
                instru_maj_data=instru_maj_data,
                instru_all_data=instru_all_data,
                dates_header=dates_header,
                basic_inputs=basic_inputs,
                patch_data=patch_data,
                major_dir=major_dir,
                major_data_type=major_data_type,
                bgn_date=bgn_date,
                calendar=calendar,
            )
    return 0
