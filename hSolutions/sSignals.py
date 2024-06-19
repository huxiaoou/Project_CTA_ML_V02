import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from rich.progress import track, Progress
from hUtils.tools import qtimer, error_handler, SFG, SFR
from hUtils.ioPlus import PySharedStackPlus, load_tsdb, CTsdbPlus
from hUtils.typeDef import TSigArgsSS
from hUtils.calendar import CCalendar
from hSolutions.sSimulation import CSimArg, CPortfolioArg

"""
Part I: Signals from single mdl
"""


class CSignal:
    def __init__(self, input_dir: str, output_dir: str, sid: str, maw: int):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sid = sid
        self.instruments: list[str] = []
        for instru_file in os.listdir(input_dir):
            instru, _ = instru_file.split(".")
            self.instruments.append(instru)
        self.maw = maw  # moving average window

    @property
    def data_settings(self) -> dict:
        return {
            "tp": np.int64,
            "trading_day": np.int32,
            self.sid: np.float32,
        }

    @property
    def data_types(self) -> np.dtype:
        return np.dtype([(k, v) for k, v in self.data_settings.items()])

    def instru_save_path(self, instru: str) -> str:
        return os.path.join(self.output_dir, f"{instru}.ss")

    def check_continuous(self, bgn_date: np.int32, calendar: CCalendar) -> bool:
        for instru in self.instruments:
            ss_path = self.instru_save_path(instru=instru)
            ss = PySharedStackPlus(ss_path, self.data_types, push_enable=True)
            if ss.check_daily_continuous(bgn_date, calendar) != 0:
                return False
        return True

    def load_input(self, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar) -> pd.DataFrame:
        base_bgn_date = calendar.get_next_date(bgn_date, -self.maw + 1)
        dfs: list[pd.DataFrame] = []
        for instru in self.instruments:
            instru_file = f"{instru}.ss"
            instru_path = os.path.join(self.input_dir, instru_file)
            ss = PySharedStackPlus(instru_path)
            instru_data = pd.DataFrame(ss.read_all())
            filter_dates = (instru_data["trading_day"] >= base_bgn_date) & (instru_data["trading_day"] <= end_date)
            instru_data = instru_data[filter_dates]
            instru_data["instru"] = instru
            dfs.append(instru_data)
        data = pd.concat(dfs, axis=0, ignore_index=True)
        data = data.sort_values(by=["tp", "trading_day", "instru", "ticker"], ascending=True)
        return data[["tp", "trading_day", "instru", "ticker", self.sid]]

    def process_nan(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna(axis=0, subset=[self.sid], how="any")

    def cal_signal(self, clean_data: pd.DataFrame) -> pd.DataFrame:
        """
        params clean_data: pd.DataFrame with columns = ["tp", "trading_day", "instru", "ticker", self.sid]
        return : pd.DataFrame with columns = ["tp", "trading_day", "instru", "ticker", self.sid]

        """
        raise NotImplementedError

    def moving_average_signal(self, signal_data: pd.DataFrame, bgn_date: np.int32) -> pd.DataFrame:
        pivot_data = pd.pivot_table(
            data=signal_data,
            index=["tp", "trading_day"],
            columns=["instru"],
            values=[self.sid],
        )
        instru_ma_data = pivot_data.fillna(0).rolling(window=self.maw).mean()
        truncated_data = instru_ma_data.reset_index("tp").truncate(before=int(bgn_date)).set_index("tp", append=True)
        normalize_data = truncated_data.div(truncated_data.abs().sum(axis=1), axis=0).fillna(0)
        stack_data = normalize_data.stack(future_stack=True).reset_index()
        return stack_data[["tp", "trading_day", "instru", self.sid]]

    def save(self, signal_data: pd.DataFrame):
        """
        params new_data: pd.DataFrame, with columns = ["tp", "trading_day", "instru", self.sid]

        """
        for instru, instru_data in signal_data.groupby(by="instru"):
            instru_new_data = instru_data.drop(labels=["instru"], axis=1)
            ss_path = self.instru_save_path(instru=instru)  # type:ignore
            ss = PySharedStackPlus(ss_path, self.data_types, push_enable=True)
            ss.append_from_DataFrame(instru_new_data, self.data_types)
        return 0

    def main(self, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar):
        if self.check_continuous(bgn_date, calendar):
            input_data = self.load_input(bgn_date, end_date, calendar)
            clean_data = self.process_nan(input_data)
            signal_data = self.cal_signal(clean_data)
            signal_data_ma = self.moving_average_signal(signal_data, bgn_date)
            self.save(signal_data_ma)
        return 0


class CSignalCrsSec(CSignal):
    @staticmethod
    def map_prediction_to_signal(data: pd.DataFrame) -> pd.DataFrame:
        n = len(data)
        s = [1] * int(n / 2) + [0] * (n % 2) + [-1] * int(n / 2)
        data["signal"] = s
        if (abs_sum := data["signal"].abs().sum()) > 0:
            data["signal"] = data["signal"] / abs_sum
        return data[["tp", "trading_day", "instru", "ticker", "signal"]]

    def cal_signal(self, clean_data: pd.DataFrame) -> pd.DataFrame:
        sorted_data = clean_data.sort_values(
            by=["tp", "trading_day", self.sid, "ticker"], ascending=[True, True, False, True]
        )
        grouped_data = sorted_data.groupby(by=["tp", "trading_day"], group_keys=False)
        signal_data = grouped_data.apply(self.map_prediction_to_signal)
        signal_data.rename(mapper={"signal": self.sid}, axis=1, inplace=True)
        return signal_data


def process_for_signal(
    input_dir: str,
    output_dir: str,
    sid: str,
    maw: int,
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
):
    signal = CSignalCrsSec(input_dir=input_dir, output_dir=output_dir, sid=sid, maw=maw)
    signal.main(bgn_date, end_date, calendar)
    return 0


@qtimer
def main_translate_signals(
    signal_args: list[TSigArgsSS],
    maw: int,
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    call_multiprocess: bool,
):
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description="[INF] Translating prediction to signals", total=len(signal_args))
            with mp.get_context("spawn").Pool() as pool:
                for input_dir, output_dir, sid in signal_args:
                    pool.apply_async(
                        process_for_signal,
                        kwds={
                            "input_dir": input_dir,
                            "output_dir": output_dir,
                            "sid": sid,
                            "maw": maw,
                            "bgn_date": bgn_date,
                            "end_date": end_date,
                            "calendar": calendar,
                        },
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for input_dir, output_dir, sid in track(signal_args, description="[INF] Translating prediction to signals"):
            process_for_signal(
                input_dir=input_dir,
                output_dir=output_dir,
                sid=sid,
                maw=maw,
                bgn_date=bgn_date,
                end_date=end_date,
                calendar=calendar,
            )
    return 0


"""
Part II: Signals from combinations of signals
"""


class CSignalPortfolio:
    TSDB_DEFAULT_BGN_DATE = np.int32(20120104)
    TSDB_DEFAULT_END_DATE = np.int32(20301231)

    def __init__(
        self,
        pid: str,
        target: str,
        weights: dict[str, float],
        portfolio_sim_args: dict[str, CSimArg],
        tsdb_root_dir: str,
        freq: str,
        prefix_user: list[str],
    ):
        self.pid = pid
        self.target = target
        self.weights = weights
        self.portfolio_sim_args = portfolio_sim_args
        self.tsdb_root_dir = tsdb_root_dir
        self.freq = freq
        self.prefix_user = prefix_user

    def load_from_tsdb(self, bgn_date: np.int32, end_date: np.int32) -> tuple[pd.DataFrame, pd.Series]:
        index_cols = ["tp", "ii", "ticker", "trading_day", "date", "time"]
        signal_data: dict[str, pd.Series] = {}
        for unique_id in self.weights:
            sim_arg = self.portfolio_sim_args[unique_id]
            unique_weight = load_tsdb(
                tsdb_root_dir=self.tsdb_root_dir,
                freq=self.freq,
                value_columns=[sim_arg.sig.tsdb_val_col],
                bgn_date=bgn_date,
                end_date=end_date,
            )
            unique_srs = unique_weight.set_index(index_cols)[sim_arg.sig.tsdb_val_col]
            signal_data[unique_id] = unique_srs
        signal_df = pd.DataFrame(signal_data)
        signal_wgt = pd.Series(self.weights)
        return signal_df, signal_wgt

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        abs_sum = df[self.target].abs().sum()
        if abs_sum > 0:
            df[self.target] = df[self.target] / abs_sum
        return df

    def cal_portfolio_weights(self, signal_df: pd.DataFrame, signal_wgt: pd.Series) -> pd.DataFrame:
        wgt_sum_data = signal_df.fillna(0) @ signal_wgt
        wgt_sum_data: pd.DataFrame = wgt_sum_data.reset_index().rename(mapper={0: self.target}, axis=1)
        wgt_sum_data_norm = wgt_sum_data.groupby(by="tp", group_keys=False).apply(self.normalize)
        return wgt_sum_data_norm

    def save_to_tsdb(self, new_data: pd.DataFrame, calendar: CCalendar):
        first_day_in_new_data = new_data["trading_day"].iloc[0]
        expected_last_trading_day = calendar.get_next_date(first_day_in_new_data, -1)

        full_name = self.prefix_user + ["signals", "portfolios", self.pid, self.target]
        new_name = ".".join(full_name)
        new_path = "/".join(full_name)
        target_tsdb_path = f"{self.tsdb_root_dir}/{self.freq[:-1]}/{self.freq[-1]}/{new_path}"
        if not os.path.exists(target_tsdb_path):
            # There are no existing data
            # load other old data('close' in this part) before first_day_in_new_data from TSDB, to use their tp and ii
            fill_data = load_tsdb(
                tsdb_root_dir=self.tsdb_root_dir,
                freq=self.freq,
                value_columns=["close"],
                bgn_date=self.TSDB_DEFAULT_BGN_DATE,
                end_date=expected_last_trading_day,
            )
            fill_data[self.target] = 0
            new_data = pd.concat([fill_data, new_data], axis=0, ignore_index=True)
            is_continuous = True
        else:
            old_data = load_tsdb(
                tsdb_root_dir=self.tsdb_root_dir,
                freq=self.freq,
                value_columns=[new_name],
                bgn_date=self.TSDB_DEFAULT_BGN_DATE,
                end_date=self.TSDB_DEFAULT_END_DATE,
            )
            last_trading_day = old_data["trading_day"].iloc[-1]
            is_continuous = last_trading_day == expected_last_trading_day

        if is_continuous:
            pytsdb = CTsdbPlus(db_root=self.tsdb_root_dir)
            pytsdb.insert_columns(tbl=self.freq, col=[self.target], df=new_data, rename={self.target: new_name})
        else:
            print(
                f"[{SFR('ERR')}] last trading day in {target_tsdb_path} is {last_trading_day}, "
                f"but expected to be {SFG(int(expected_last_trading_day))} "
            )
        return 0

    def main(self, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar):
        signal_df, signal_wgt = self.load_from_tsdb(bgn_date, end_date)
        wgt_sum_data_norm = self.cal_portfolio_weights(signal_df, signal_wgt)
        self.save_to_tsdb(wgt_sum_data_norm, calendar)
        return 0


def process_for_cal_signal_portfolio(
    portfolio_arg: CPortfolioArg,
    tsdb_root_dir: str,
    freq: str,
    prefix_user: list[str],
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
):
    signal_portfolio = CSignalPortfolio(
        pid=portfolio_arg.pid,
        target=portfolio_arg.target,
        weights=portfolio_arg.weights,
        portfolio_sim_args=portfolio_arg.portfolio_sim_args,
        tsdb_root_dir=tsdb_root_dir,
        freq=freq,
        prefix_user=prefix_user,
    )
    signal_portfolio.main(bgn_date, end_date, calendar)
    return 0


@qtimer
def main_translate_signals_portfolio(
    portfolio_args: list[CPortfolioArg],
    tsdb_root_dir: str,
    freq: str,
    prefix_user: list[str],
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    call_multiprocess: bool,
):
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description="[INF] Combing portfolio signals", total=len(portfolio_args))
            with mp.get_context("spawn").Pool() as pool:
                for portfolio_arg in portfolio_args:
                    pool.apply_async(
                        process_for_cal_signal_portfolio,
                        kwds={
                            "portfolio_arg": portfolio_arg,
                            "tsdb_root_dir": tsdb_root_dir,
                            "freq": freq,
                            "prefix_user": prefix_user,
                            "bgn_date": bgn_date,
                            "end_date": end_date,
                            "calendar": calendar,
                        },
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for portfolio_arg in track(portfolio_args, description="[INF] Combing portfolio signals"):
            process_for_cal_signal_portfolio(
                portfolio_arg=portfolio_arg,
                tsdb_root_dir=tsdb_root_dir,
                freq=freq,
                prefix_user=prefix_user,
                bgn_date=bgn_date,
                end_date=end_date,
                calendar=calendar,
            )
    return 0
