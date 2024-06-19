import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from rich.progress import Progress
from hUtils.tools import qtimer, batched, error_handler, SFG
from hUtils.ioPlus import load_tsdb, PySharedStackPlus
from hUtils.instruments import parse_instrument_from_contract
from hUtils.calendar import CCalendar


def add_instru(raw_data: pd.DataFrame, ticker_name: str = "ticker"):
    """
    param raw_data: a pd.DataFrame with a column named ticker_name
    """
    raw_data["instru"] = raw_data[ticker_name].map(lambda z: parse_instrument_from_contract(z.decode("utf-8")))
    return 0


def process_for_slice(
    tsdb_root_dir: str, freq: str, value_columns: list[str], sliced_bgn_date: np.int32, sliced_end_date: np.int32
) -> pd.DataFrame:
    raw_data = load_tsdb(
        tsdb_root_dir=tsdb_root_dir,
        freq=freq,
        value_columns=value_columns,
        bgn_date=sliced_bgn_date,
        end_date=sliced_end_date,
    )
    add_instru(raw_data)
    return raw_data[["tp", "date", "time", "trading_day", "ticker", "instru"] + value_columns]


def load_from_tsdb(
    iter_dates: list[np.int32], batch_size: int, tsdb_root_dir: str, freq: str, value_columns: list[str]
) -> pd.DataFrame:
    with Progress() as pb:
        total = int(np.ceil(len(iter_dates) / batch_size))
        main_task = pb.add_task(
            description=f"[INF] Generating minute-bar from {SFG(tsdb_root_dir)} with freq = {SFG(freq)}",
            total=total,
        )
        jobs = []
        with mp.get_context("spawn").Pool() as pool:
            for sliced_dates in batched(iter_dates, batch_size=batch_size):
                sliced_bgn_date, sliced_end_date = sliced_dates[0], sliced_dates[-1]
                print(f"[INF] Slice is {SFG(sliced_bgn_date)} -> {SFG(sliced_end_date)}")
                job = pool.apply_async(
                    process_for_slice,
                    kwds={
                        "tsdb_root_dir": tsdb_root_dir,
                        "freq": freq,
                        "value_columns": value_columns,
                        "sliced_bgn_date": sliced_bgn_date,
                        "sliced_end_date": sliced_end_date,
                    },
                    callback=lambda _: pb.update(main_task, advance=1),
                    error_callback=error_handler,
                )
                jobs.append(job)
            pool.close()
            pool.join()
        dfs: list[pd.DataFrame] = [job.get() for job in jobs]
        minute_data = pd.concat(dfs, axis=0, ignore_index=True)
        return minute_data


def fillna_for_values(data: pd.DataFrame, values: dict) -> pd.DataFrame:
    for k, v in values.items():
        if v in [np.int32, np.int64]:
            data[k] = data[k].fillna(0)
    return data


def drop_invalid(data: pd.DataFrame, vars_to_drop_invalid: list[str]) -> pd.DataFrame:
    return data.dropna(axis=0, how="all", subset=vars_to_drop_invalid)


def process_for_saving_instru(
    instru: str,
    instru_data: pd.DataFrame,
    minute_bar_dir: str,
    freq: str,
    new_data_type: np.dtype,
    calendar: CCalendar,
):
    new_data = instru_data.drop(labels="instru", axis=1).sort_values("tp", ascending=True)
    ss_file = f"{instru}.ss"
    ss_path = os.path.join(minute_bar_dir, freq, ss_file)
    ss = PySharedStackPlus(ss_path, dtype=new_data_type, push_enable=True)
    if ss.check_daily_continuous(new_data["trading_day"].iloc[0], calendar) == 0:
        ss.append_from_DataFrame(new_data=new_data, new_data_type=new_data_type)
    return 0


def saving_to_ss(
    minute_data: pd.DataFrame, minute_bar_dir: str, freq: str, new_data_type: np.dtype, calendar: CCalendar
):
    with Progress() as pb:
        grouped_data = minute_data.groupby(by="instru")
        main_task = pb.add_task(description="[INF] Saving instruments", total=len(grouped_data))
        with mp.get_context("spawn").Pool() as pool:
            for instru, instru_data in grouped_data:
                pool.apply_async(
                    process_for_saving_instru,
                    kwds={
                        "instru": instru,
                        "instru_data": instru_data,
                        "minute_bar_dir": minute_bar_dir,
                        "freq": freq,
                        "new_data_type": new_data_type,
                        "calendar": calendar,
                    },
                    callback=lambda _: pb.update(main_task, advance=1),
                    error_callback=error_handler,
                )
            pool.close()
            pool.join()
    return 0


@qtimer
def main_min_bar(
    bgn_date: np.int32,
    end_date: np.int32,
    tsdb_root_dir: str,
    freq: str,
    values: dict,
    vars_to_drop_invalid: list[str],
    minute_bar_dir: str,
    calendar: CCalendar,
    batch_size: int,
):
    iter_dates = calendar.get_iter_list(bgn_date, end_date)
    value_columns = list(values)
    new_data_type = np.dtype(
        [
            ("tp", np.int64),
            ("date", np.int32),
            ("time", np.int32),
            ("trading_day", np.int32),
            ("ticker", "S8"),
        ]
        + [(k, v) for k, v in values.items()]
    )
    minute_data = load_from_tsdb(iter_dates, batch_size, tsdb_root_dir, freq, value_columns)
    minute_data = fillna_for_values(minute_data, values)
    minute_data = drop_invalid(minute_data, vars_to_drop_invalid)
    saving_to_ss(minute_data, minute_bar_dir, freq, new_data_type, calendar)
    return 0
