import os
import datetime as dt
import numpy as np
import pandas as pd
import multiprocessing as mp
from rich.progress import track, Progress
from hUtils.tools import qtimer, SFG, error_handler, batched
from hUtils.calendar import CCalendar
from hUtils.instruments import CInstrumentInfoTable, parse_instrument_from_contract
from hUtils.wds import CDownloadEngineWDS
from hUtils.ioPlus import PySharedStackPlus


def reformat(raw_data: pd.DataFrame, instru_info_tab: CInstrumentInfoTable, selected_cols: list[str]) -> pd.DataFrame:
    def __trans_wind_code_to_ticker(z: pd.Series):
        wind_code, trade_date = z["S_INFO_WINDCODE"], z["TRADE_DT"]
        ticker, exchange = wind_code.split(".")
        if exchange in ["SHF", "INE", "DCE", "GFE"]:
            ticker = ticker.lower()
        instrument = parse_instrument_from_contract(ticker)
        ticker = instru_info_tab.fix_contract_id_alt(ticker, exchange, instrument, trade_date)
        return ticker.encode("utf-8"), instrument

    raw_data["ticker"], raw_data["instrument"] = zip(*raw_data.apply(__trans_wind_code_to_ticker, axis=1))
    raw_data["tp"] = raw_data["TRADE_DT"].map(
        lambda z: dt.datetime.strptime(f"{z} 16:00:00", "%Y%m%d %H:%M:%S").timestamp() * 1e9
    )
    raw_data["trading_day"] = raw_data["TRADE_DT"].map(lambda z: np.int32(z))
    raw_data["FS_INFO_MEMBERNAME"] = raw_data["FS_INFO_MEMBERNAME"].map(lambda z: z.encode("utf-8"))
    reformat_data = raw_data[["tp", "trading_day", "ticker"] + selected_cols + ["instrument"]]
    return reformat_data


def process_download_and_reformat(
    bgn_date: np.int32,
    end_date: np.int32,
    download_values: list[str],
    wds: CDownloadEngineWDS,
    instru_info_tab: CInstrumentInfoTable,
    futures_type: str,
    verbose: bool,
) -> pd.DataFrame:
    raw_data = wds.download_futures_positions_by_dates(
        bgn_date=bgn_date,
        end_date=end_date,
        download_values=download_values,
        futures_type=futures_type,
    )
    reformat_data = reformat(raw_data, instru_info_tab, selected_cols=download_values[:-1])
    if verbose:
        print(
            f"[INF] MbrPos Data between [{SFG(str(bgn_date))}, {SFG(str(end_date))}] "
            f"of {SFG(futures_type)} are processed"
        )
    return reformat_data


@qtimer
def download_reformat_and_concat(
    bgn_date: np.int32,
    end_date: np.int32,
    batch_size: int,
    download_values: list[str],
    wds: CDownloadEngineWDS,
    calendar: CCalendar,
    instru_info_tab: CInstrumentInfoTable,
    call_multiprocess: bool,
    processes: int,
    verbose: bool,
) -> pd.DataFrame:
    iter_dates = calendar.get_iter_list(bgn_date, end_date)
    batch_dates_ary = list(batched(iterable=iter_dates, batch_size=batch_size))
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(
                description="[INF] Downloading and Reformatting", total=len(batch_dates_ary) * 2
            )  # *2 for E and C
            with mp.get_context("spawn").Pool(processes) as pool:
                jobs = []
                for batch_dates in batch_dates_ary:
                    b, e = batch_dates[0], batch_dates[-1]
                    job = pool.apply_async(
                        process_download_and_reformat,
                        args=(b, e, download_values, wds, instru_info_tab, "C", verbose),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                    jobs.append(job)
                    job = pool.apply_async(
                        process_download_and_reformat,
                        args=(b, e, download_values, wds, instru_info_tab, "E", verbose),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                    jobs.append(job)
                pool.close()
                pool.join()
            dfs: list[pd.DataFrame] = [job.get() for job in jobs]
    else:
        dfs: list[pd.DataFrame] = []
        for batch_dates in track(batch_dates_ary, description="[INF] Downloading and Reformatting"):
            b, e = batch_dates[0], batch_dates[-1]
            dfs.append(process_download_and_reformat(b, e, download_values, wds, instru_info_tab, "C", verbose))
            dfs.append(process_download_and_reformat(b, e, download_values, wds, instru_info_tab, "E", verbose))
    downloaded_data = pd.concat(dfs, axis=0, ignore_index=True)
    downloaded_data.sort_values(by=["instrument", "trading_day", "ticker"], ascending=True, inplace=True)
    return downloaded_data


def process_save(
    instru: str,
    instru_data: pd.DataFrame,
    mbr_pos_dir: str,
    mbr_pos_data_type: np.dtype,
    bgn_date: np.int32,
    calendar: CCalendar,
    verbose: bool,
):
    mbr_pos_data = instru_data.drop(labels="instrument", axis=1)
    mbr_pos_file = f"{instru}.ss"
    mbr_pos_path = os.path.join(mbr_pos_dir, mbr_pos_file)
    ss = PySharedStackPlus(mbr_pos_path, dtype=mbr_pos_data_type, push_enable=True)
    if ss.check_daily_continuous(coming_next_date=bgn_date, calendar=calendar) <= 0:
        # member position can be missed
        if verbose:
            print(f"[INF] begin to save member position data for {SFG(instru)} ...")
        ss.append_from_DataFrame(new_data=mbr_pos_data, new_data_type=mbr_pos_data_type)
    return 0


@qtimer
def save(
    downloaded_data: pd.DataFrame,
    bgn_date: np.int32,
    mbr_pos_dir: str,
    mbr_pos_data_type: np.dtype,
    calendar: CCalendar,
    call_multiprocess: bool,
    processes: int,
    verbose: bool,
):
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(
                description="[INF] Saving member position data",
                total=len(downloaded_data["instrument"].unique()),
            )
            with mp.get_context("spawn").Pool(processes) as pool:
                for instru, instru_data in downloaded_data.groupby(by="instrument"):
                    pool.apply_async(
                        process_save,
                        args=(instru, instru_data, mbr_pos_dir, mbr_pos_data_type, bgn_date, calendar, verbose),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for instru, instru_data in track(
            downloaded_data.groupby(by="instrument"),
            description="[INF] Saving member position data",
        ):
            process_save(
                instru=instru,  # type:ignore
                instru_data=instru_data,
                mbr_pos_dir=mbr_pos_dir,
                mbr_pos_data_type=mbr_pos_data_type,
                bgn_date=bgn_date,
                calendar=calendar,
                verbose=verbose,
            )
    return 0


@qtimer
def main_mbr_pos(
    bgn_date: np.int32,
    end_date: np.int32,
    download_values: list[str],
    wds: CDownloadEngineWDS,
    calendar: CCalendar,
    instru_info_tab: CInstrumentInfoTable,
    mbr_pos_dir: str,
    batch_size: int,
    call_multiprocess: bool,
    processes: int,
    verbose: bool,
):
    pd.set_option("display.unicode.east_asian_width", True)
    mbr_pos_data_type = np.dtype(
        [
            ("tp", np.int64),
            ("trading_day", np.int32),
            ("ticker", "S8"),
            ("info_type", np.int32),
            ("info_rank", np.int32),
            ("member", "S24"),
            ("info_qty", np.int32),
            ("info_qty_dlt", np.int32),
        ]
    )
    downloaded_data = download_reformat_and_concat(
        bgn_date=bgn_date,
        end_date=end_date,
        batch_size=batch_size,
        download_values=download_values,
        wds=wds,
        calendar=calendar,
        instru_info_tab=instru_info_tab,
        call_multiprocess=call_multiprocess,
        processes=processes,
        verbose=verbose,
    )
    save(
        downloaded_data=downloaded_data,
        bgn_date=bgn_date,
        mbr_pos_dir=mbr_pos_dir,
        mbr_pos_data_type=mbr_pos_data_type,
        calendar=calendar,
        call_multiprocess=False,  # Yes, not calling multiprocess would be faster, reasons unknown, guess this task is io-intensive
        processes=processes,
        verbose=verbose,
    )
    return 0
