import os
import numpy as np
import pandas as pd
import scipy.stats as sps
import multiprocessing as mp
from rich.progress import track, Progress
from hUtils.tools import qtimer, SFG, error_handler
from hUtils.ioPlus import PySharedStackPlus
from hUtils.calendar import CCalendar
from hUtils.typeDef import TFactorClass, TFactorNames, TFactorName


class CFactorGeneric:
    def __init__(self, factor_class: TFactorClass, factor_names: TFactorNames, save_by_instru_dir: str):
        self.factor_class = factor_class
        self.factor_names = factor_names
        self.save_by_instru_dir: str = save_by_instru_dir

    def get_factor_by_instru_path(self, instru: str):
        factor_by_instru_file = f"{instru}.ss"
        factor_by_instru_path = os.path.join(self.save_by_instru_dir, self.factor_class, factor_by_instru_file)
        return factor_by_instru_path

    def get_factor_save_dtype(self) -> np.dtype:
        fix_cols = [("tp", np.int64), ("trading_day", np.int32), ("ticker", "S8")]
        fac_cols = [(f, np.float64) for f in self.factor_names]
        return np.dtype(fix_cols + fac_cols)

    def save_by_instru(self, factor_data: pd.DataFrame, instru: str, calendar: CCalendar):
        """
        params: factor_data, columns = ["tp", "trading_day", "ticker"] + self.factor_names

        """
        factor_by_instru_dtype = self.get_factor_save_dtype()
        factor_by_instru_path = self.get_factor_by_instru_path(instru)
        ss = PySharedStackPlus(factor_by_instru_path, dtype=factor_by_instru_dtype, push_enable=True)
        if ss.check_daily_continuous(factor_data["trading_day"].iloc[0], calendar) == 0:
            ss.append_from_DataFrame(factor_data, factor_by_instru_dtype)
        return 0


class CFactor(CFactorGeneric):
    def __init__(
        self,
        factor_class: TFactorClass,
        factor_names: TFactorNames,
        universe: dict[str, dict[str, str]],
        factors_by_instru_dir: str,
        minute_bar_dir: str | None = None,
        major_dir: str | None = None,
        preprocess_dir: str | None = None,
        mbr_pos_dir: str | None = None,
        alternative_dir: str | None = None,
        market_dir: str | None = None,
    ):
        self.universe = universe
        self.minute_bar_dir = minute_bar_dir
        self.major_dir = major_dir
        self.preprocess_dir = preprocess_dir
        self.mbr_pos_dir = mbr_pos_dir
        self.alternative_dir = alternative_dir
        self.market_dir = market_dir
        super().__init__(factor_class, factor_names, save_by_instru_dir=factors_by_instru_dir)

    def load_minute_bar(self, instru: str, freq: str) -> pd.DataFrame:
        minute_bar_file = f"{instru}.ss"
        if self.minute_bar_dir:
            minute_bar_path = os.path.join(self.minute_bar_dir, freq, minute_bar_file)
        else:
            raise ValueError("Argument minute_bar_dir must be provided")
        ss = PySharedStackPlus(minute_bar_path)
        return pd.DataFrame(ss.read_all())

    def load_major(self, instru: str) -> pd.DataFrame:
        major_file = f"{instru}.ss"
        if self.major_dir:
            major_path = os.path.join(self.major_dir, major_file)
        else:
            raise ValueError("Argument major_dir must be provided")
        ss = PySharedStackPlus(major_path)
        return pd.DataFrame(ss.read_all())

    def load_preprocess(self, instru: str) -> pd.DataFrame:
        preprocess_file = f"{instru}.ss"
        if self.preprocess_dir:
            preprocess_path = os.path.join(self.preprocess_dir, preprocess_file)
        else:
            raise ValueError("Argument preprocess_dir must be provided")
        ss = PySharedStackPlus(preprocess_path)
        df = pd.DataFrame(ss.read_all())
        return df

    def load_mbr_pos(self, instru: str) -> pd.DataFrame:
        mbr_pos_file = f"{instru}.ss"
        if self.mbr_pos_dir:
            mbr_pos_path = os.path.join(self.mbr_pos_dir, mbr_pos_file)
        else:
            raise ValueError("Argument mbr_pos_dur must provided")
        ss = PySharedStackPlus(mbr_pos_path)
        df = pd.DataFrame(ss.read_all())
        return df

    def load_forex(self) -> pd.DataFrame:
        forex_file = "forex.ss"
        if self.alternative_dir:
            forex_path = os.path.join(self.alternative_dir, forex_file)
        else:
            raise ValueError("Argument alternative_dir must be provided")
        ss = PySharedStackPlus(forex_path)
        df = pd.DataFrame(ss.read_all())
        return df

    def load_macro(self) -> pd.DataFrame:
        forex_file = "macro.ss"
        if self.alternative_dir:
            forex_path = os.path.join(self.alternative_dir, forex_file)
        else:
            raise ValueError("Argument alternative_dir must be provided")
        ss = PySharedStackPlus(forex_path)
        df = pd.DataFrame(ss.read_all())
        return df

    def load_market(self) -> pd.DataFrame:
        market_file = "market.ss"
        if self.market_dir:
            market_path = os.path.join(self.market_dir, market_file)
        else:
            raise ValueError("Argument market_dir must be provided")
        ss = PySharedStackPlus(market_path)
        df = pd.DataFrame(ss.read_all())
        return df

    @staticmethod
    def truncate(data: pd.DataFrame, index: str, before: np.int32, after: np.int32) -> pd.DataFrame:
        return data.set_index(index).truncate(before=int(before), after=int(after))

    def get_adj_minute_bar_data(
        self, instru: str, win_start_date: np.int32, end_date: np.int32, freq: str
    ) -> pd.DataFrame:
        minute_bar_data = self.load_minute_bar(instru, freq)
        adj_minute_bar_data = self.truncate(minute_bar_data, index="trading_day", before=win_start_date, after=end_date)
        return adj_minute_bar_data

    def get_adj_major_data(
        self, instru: str, win_start_date: np.int32, end_date: np.int32, y: str = ""
    ) -> pd.DataFrame:
        major_data = self.load_major(instru=instru)
        adj_major_data = self.truncate(major_data, index="trading_day", before=win_start_date, after=end_date)
        if y:
            adj_major_data[y] = adj_major_data[y] * 100
        return adj_major_data

    def get_adj_preprocess_data(self, instru: str, win_start_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        preprocess_data = self.load_preprocess(instru=instru)
        adj_preprocess_data = self.truncate(preprocess_data, index="trading_day", before=win_start_date, after=end_date)
        return adj_preprocess_data

    def get_adj_mbr_pos_data(self, instru: str, win_start_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        mbr_pos_data = self.load_mbr_pos(instru=instru)
        adj_mbr_pos_data = self.truncate(mbr_pos_data, index="trading_day", before=win_start_date, after=end_date)
        return adj_mbr_pos_data

    def get_adj_macro_data(self, win_start_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        macro_data = self.load_macro()
        adj_macro_data = self.truncate(macro_data, index="trading_day", before=win_start_date, after=end_date)
        return adj_macro_data

    def get_adj_market_data(self, win_start_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        market_data = self.load_market()
        adj_market_data = self.truncate(market_data, index="trading_day", before=win_start_date, after=end_date)
        return adj_market_data

    def get_adj_forex_data(self, win_start_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        forex_data = self.load_forex()
        adj_forex_data = self.truncate(forex_data, index="trading_day", before=win_start_date, after=end_date)
        return adj_forex_data

    def get_factor_data(self, input_data: pd.DataFrame, bgn_date: np.int32) -> pd.DataFrame:
        input_data = input_data.truncate(before=int(bgn_date)).reset_index()
        factor_data = input_data[["tp", "trading_day", "ticker"] + self.factor_names]
        return factor_data

    def cal_factor_by_instru(
        self,
        instru: str,
        bgn_date: np.int32,
        end_date: np.int32,
        calendar: CCalendar,
    ) -> pd.DataFrame:
        """
        This function is to be realized by specific factors

        :return : a pd.DataFrame with first 3 columns must be = ["tp", "trading_day", "ticker"]
                  then followed by factor names
        """
        raise NotImplementedError()

    def process_by_instru(self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar):
        factor_data = self.cal_factor_by_instru(instru, bgn_date, end_date, calendar)
        self.save_by_instru(factor_data, instru, calendar)
        return 0

    # @qtimer
    def main(
        self,
        bgn_date: np.int32,
        end_date: np.int32,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
    ):
        if call_multiprocess:
            with Progress() as pb:
                main_task = pb.add_task(
                    description=f"[INF] Calculating factor {SFG(self.factor_class)}", total=len(self.universe)
                )
                with mp.get_context("spawn").Pool(processes) as pool:
                    for instru in self.universe:
                        pool.apply_async(
                            self.process_by_instru,
                            args=(instru, bgn_date, end_date, calendar),
                            callback=lambda _: pb.update(main_task, advance=1),
                            error_callback=error_handler,
                        )
                    pool.close()
                    pool.join()
        else:
            for instru in track(self.universe, description=f"[INF] Calculating factor {SFG(self.factor_class)}"):
                # for instru in self.universe:
                self.process_by_instru(instru, bgn_date, end_date, calendar)
        return 0


def trans_raw_to_rank(raw_data: pd.DataFrame, ref_cols: list[str]) -> pd.DataFrame:
    target = raw_data[ref_cols]
    res = target.rank() / (target.count() + 1)
    return res


class CFactorNeu(CFactorGeneric):
    def __init__(
        self,
        ref_factor: CFactorGeneric,
        universe: dict[str, dict[str, str]],
        major_dir: str,
        available_dir: str,
        neutral_by_instru_dir: str,
    ):
        self.ref_factor: CFactorGeneric = ref_factor
        self.universe = universe
        self.major_dir = major_dir
        self.available_dir = available_dir
        super().__init__(
            factor_class=TFactorClass(f"{self.ref_factor.factor_class}-NEU"),
            factor_names=TFactorNames([TFactorName(f"{z}-NEU") for z in self.ref_factor.factor_names]),
            save_by_instru_dir=neutral_by_instru_dir,
        )

    def load_ref_factor_by_instru(self, instru: str) -> pd.DataFrame:
        factor_by_instru_file = f"{instru}.ss"
        factor_by_instru_path = os.path.join(
            self.ref_factor.save_by_instru_dir, self.ref_factor.factor_class, factor_by_instru_file
        )
        ss = PySharedStackPlus(pth=factor_by_instru_path)
        df = pd.DataFrame(ss.read_all())
        return df

    def load_ref_factor(self, bgn_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        ref_dfs: list[pd.DataFrame] = []
        for instru in self.universe:
            df = self.load_ref_factor_by_instru(instru)
            df = df.set_index("trading_day").truncate(before=int(bgn_date), after=int(end_date))
            df["instrument"] = instru
            ref_dfs.append(df)
        res = pd.concat(ref_dfs, axis=0, ignore_index=False)
        res = res.reset_index().sort_values(by=["trading_day"], ascending=True)
        res = res[["tp", "trading_day", "instrument"] + self.ref_factor.factor_names]
        return res

    def load_available(self, bgn_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        available_file = "available.ss"
        available_path = os.path.join(self.available_dir, available_file)
        ss = PySharedStackPlus(available_path)
        df = pd.DataFrame(ss.read_all())
        df = df.set_index("trading_day").truncate(before=int(bgn_date), after=int(end_date)).reset_index()
        df["instrument"] = df["instrument"].map(lambda z: z.decode("utf-8"))
        df = df[["tp", "trading_day", "instrument", "sectorL1"]]
        return df

    def normalize(self, data: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
        jobs = []
        with Progress() as pb:
            grouped_data = data.groupby(group_keys)
            main_task = pb.add_task(description="[INF] Normalizing by date", total=len(grouped_data))
            with mp.get_context("spawn").Pool(48) as pool:
                for _, sub_df in grouped_data:
                    job = pool.apply_async(
                        trans_raw_to_rank,
                        args=(sub_df, self.ref_factor.factor_names),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                    jobs.append(job)
                pool.close()
                pool.join()
            dfs: list[pd.DataFrame] = [job.get() for job in jobs]
            rank_df = pd.concat(dfs, axis=0, ignore_index=False)
            normalize_df = pd.DataFrame(
                data=sps.norm.ppf(rank_df.values),
                index=rank_df.index,
                columns=self.factor_names,
            )
        return normalize_df

    def neutralize_by_date(self, net_ref_factor_data: pd.DataFrame) -> pd.DataFrame:
        normalize_df = self.normalize(net_ref_factor_data, group_keys=["trading_day", "sectorL1"])
        if (s0 := len(normalize_df)) != (s1 := len(net_ref_factor_data)):
            raise ValueError(f"[ERR] Size after normalization = {s0} != Size before normalization {s1}")
        else:
            merge_df = net_ref_factor_data[["tp", "trading_day", "instrument"]].merge(
                right=normalize_df[self.factor_names],
                left_index=True,
                right_index=True,
                how="left",
            )
            res = merge_df[["tp", "trading_day", "instrument"] + self.factor_names]
            return res

    def load_header(self, instru: str, bgn_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        major_file = f"{instru}.ss"
        major_path = os.path.join(self.major_dir, major_file)
        ss = PySharedStackPlus(major_path)
        df = pd.DataFrame(ss.read_all())
        df = df.set_index("trading_day").truncate(before=int(bgn_date), after=int(end_date)).reset_index()
        return df[["tp", "trading_day", "ticker"]]

    def process_by_instru(
        self,
        instru: str,
        instru_neu_factor_data: pd.DataFrame,
        bgn_date: np.int32,
        end_date: np.int32,
        calendar,
    ):
        instru_header = self.load_header(instru, bgn_date, end_date)
        instru_data = pd.merge(
            left=instru_header,
            right=instru_neu_factor_data,
            on=["tp", "trading_day"],
            how="left",
        )
        factor_data = instru_data[["tp", "trading_day", "ticker"] + self.factor_names]
        self.save_by_instru(factor_data, instru, calendar)
        return 0

    # @qtimer
    def main_neutralize(
        self,
        bgn_date: np.int32,
        end_date: np.int32,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
    ):
        ref_factor_data = self.load_ref_factor(bgn_date, end_date)
        available_data = self.load_available(bgn_date, end_date)
        net_ref_factor_data = pd.merge(
            left=available_data,
            right=ref_factor_data,
            on=["tp", "trading_day", "instrument"],
            how="left",
        ).sort_values(by=["trading_day", "sectorL1"])
        neu_factor_data = self.neutralize_by_date(net_ref_factor_data)
        if call_multiprocess:
            with mp.get_context("spawn").Pool(processes) as pool:
                for instru, instru_neu_factor_data in neu_factor_data.groupby(by="instrument"):
                    pool.apply_async(
                        self.process_by_instru,
                        args=(instru, instru_neu_factor_data, bgn_date, end_date, calendar),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
        else:
            for instru, instru_neu_factor_data in track(
                neu_factor_data.groupby(by="instrument"),
                description="[INF] Neutralizing raw factors",
            ):
                self.process_by_instru(instru, instru_neu_factor_data, bgn_date, end_date, calendar)  # type:ignore
        return 0
