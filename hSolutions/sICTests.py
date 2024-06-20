import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from dataclasses import dataclass
from itertools import product
from rich.progress import track, Progress
from hUtils.tools import qtimer, SFG, SFY, SFR, error_handler
from hUtils.calendar import CCalendar
from hUtils.instruments import parse_instrument_from_contract
from hUtils.ioPlus import PySharedStackPlus, load_tsdb
from hUtils.plot import plot_lines
from hSolutions.sFactorAlg import CCfgFactor
from hUtils.typeDef import TPrefix, TFactorClass, TFactorName, TReturnClass, TReturnName, TReturnNames

"""
Part I: Calculate IC
"""


# def cal_ic(data: pd.DataFrame, x: str, y: str) -> float:
#     if len(data) == 1:
#         return 0.0
#     else:
#         return data[[x, y]].corr(method="spearman").at[x, y]


def gen_weight(n: int) -> np.ndarray:
    k, d = n // 2, n % 2
    rou = np.power(0.5, 1 / k)
    if d == 0:
        s = np.array(list(range(k)) + list(range(k - 1, -1, -1)))
    else:
        s = np.array(list(range(k)) + [k] + list(range(k - 1, -1, -1)))
    w = np.power(rou, s)
    return w / np.sum(w)


def cal_wgt_cor(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    xm, ym = x @ w, y @ w
    v_xy = (x * y) @ w - xm * ym
    v_xx = (x * x) @ w - xm * xm
    v_yy = (y * y) @ w - ym * ym
    if (d := v_xx * v_yy) > 1e-12:
        return v_xy / np.sqrt(d)
    else:
        return 0


def cal_ic(data: pd.DataFrame, x: str, y: str) -> float:
    if (n := len(data)) <= 1:
        return 0.0
    else:
        sorted_data = data[[x, y]].sort_values(by=x)
        w = gen_weight(n)
        xv, yv = sorted_data[x].values, sorted_data[y].values
        return cal_wgt_cor(xv, yv, w)  # type:ignore


@dataclass(frozen=True)
class CIcFac:
    fac_prefix: TPrefix
    fac_class: TFactorClass
    fac_name: TFactorName

    @property
    def class_and_name(self) -> str:
        return f"{self.fac_class}.{self.fac_name}"

    @property
    def prefix_str(self) -> str:
        return ".".join(self.fac_prefix)


@dataclass(frozen=True)
class CIcRet:
    ret_prefix: TPrefix
    ret_class: TReturnClass
    ret_name: TReturnName

    @property
    def class_and_name(self) -> str:
        return f"{self.ret_class}.{self.ret_name}"

    @property
    def prefix_str(self) -> str:
        return ".".join(self.ret_prefix)


class CICTests:
    def __init__(
        self,
        ic_fac: CIcFac,
        ic_ret: CIcRet,
        shift: int,
        tsdb_root_dir: str,
        available_dir: str,
        ic_save_dir: str,
        freq: str,
        sectors: list[str],
    ):
        """
        params fac_prefix: ["team", "huxo", "neutral_by_instru"]
        params fac_class: "CV-D240-NEU"
        params fac_name: "CV180-D240-NEU"

        params ret_prefix: ["team", "huxo", "Y"]
        params ret_class: "001L1-NEU"
        params ret_name: "CloseRtn001L1-NEU"
        """

        self.ic_fac = ic_fac
        self.ic_ret = ic_ret
        self.shift = shift
        self.tsdb_root_dir = tsdb_root_dir
        self.available_dir = available_dir
        self.ic_save_dir = ic_save_dir
        self.freq = freq
        self.sectors = sectors

    @property
    def fac_val_col(self) -> str:
        return f"{self.ic_fac.prefix_str}.{self.ic_fac.class_and_name}"

    @property
    def ret_val_col(self) -> str:
        return f"{self.ic_ret.prefix_str}.{self.ic_ret.class_and_name}"

    def load_available(self, bgn_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        available_file = "available.ss"
        available_path = os.path.join(self.available_dir, available_file)
        ss = PySharedStackPlus(available_path)
        df = pd.DataFrame(ss.read_all())
        df = df.set_index("trading_day").truncate(before=int(bgn_date), after=int(end_date)).reset_index()
        df["instrument"] = df["instrument"].map(lambda z: z.decode("utf-8"))
        df = df[["tp", "trading_day", "instrument", "sectorL1"]]
        return df

    @staticmethod
    def reformat(raw_data: pd.DataFrame, rename_mapper: dict[str, str], selected_cols: list[str]) -> pd.DataFrame:
        rft_data = raw_data.rename(mapper=rename_mapper, axis=1)
        return rft_data[selected_cols]

    def check_fac_and_ret(self, rft_fac_data: pd.DataFrame, rft_ret_data: pd.DataFrame) -> bool:
        if (s0 := len(rft_fac_data)) != (s1 := len(rft_ret_data)):
            print(f"[INF] Size of factor  data = {SFY(s0)}")
            print(f"[INF] Size of return  data = {SFY(s1)}")
            print(f"[{SFR('ERR')}] Size between factor and return of {self.ic_fac.class_and_name} does not match.")
            return False
        else:
            return True

    @staticmethod
    def merge_fac_and_ret(rft_fac_data: pd.DataFrame, rft_ret_data: pd.DataFrame) -> pd.DataFrame:
        merged_data = pd.merge(left=rft_fac_data, right=rft_ret_data, on=["tp", "trading_day", "ticker"], how="inner")
        merged_data["instrument"] = merged_data["ticker"].map(
            lambda z: parse_instrument_from_contract(z.decode("utf-8"))
        )
        return merged_data

    def merge_to_avlb(self, available_data: pd.DataFrame, rft_fac_and_ret_data: pd.DataFrame) -> pd.DataFrame:
        avlb_fac_and_ret_data = (
            available_data.merge(
                right=rft_fac_and_ret_data,
                on=["tp", "trading_day", "instrument"],
                how="left",
            )
            .sort_values(by=["trading_day", "sectorL1", "instrument"])
            .dropna(axis=0, subset=[self.ic_fac.fac_name])
        )
        return avlb_fac_and_ret_data

    def get_ic(self, avlb_fac_and_ret_data: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        grouped_data = avlb_fac_and_ret_data.groupby(by=["tp", "trading_day", "sectorL1"])
        ic_data_grouped: pd.DataFrame = grouped_data.apply(
            cal_ic, x=self.ic_fac.fac_name, y=self.ic_ret.ret_name  # type:ignore
        )
        ic_data = pd.pivot_table(
            data=ic_data_grouped.reset_index(),
            index=["tp", "trading_day"],
            columns="sectorL1",
            values=0,
            aggfunc=pd.Series.mean,
        )
        ic_data.columns = [s.decode("utf-8") for s in ic_data.columns]  # type:ignore
        for s in self.sectors:
            if s not in ic_data.columns:
                ic_data[s] = 0
                if verbose:
                    print(
                        f"[INF] IC data of {SFY(s)} is not available for "
                        f"{SFY(self.ic_fac.class_and_name)} and {SFY(self.ic_ret.class_and_name)}"
                    )
        ic_data = ic_data[self.sectors].fillna(0)
        return ic_data

    def save(self, new_data: pd.DataFrame, new_data_type: np.dtype):
        ss = PySharedStackPlus(os.path.join(self.ic_save_dir, self.ic_ret.ret_class, self.ic_ret.ret_name))
        ss.append_from_DataFrame(new_data=new_data, new_data_type=new_data_type)
        return 0

    def main_ic_test(self, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar, verbose: bool):
        base_bgn_date = calendar.get_next_date(bgn_date, -self.shift)
        base_end_date = calendar.get_next_date(end_date, -self.shift)

        new_data_type = np.dtype(
            [("tp", np.int64), ("trading_day", np.int32)] + [(s, np.float32) for s in self.sectors]
        )
        ss_path = os.path.join(self.ic_save_dir, self.ic_ret.ret_name, f"{self.ic_fac.fac_name}.ss")
        ss = PySharedStackPlus(ss_path, dtype=new_data_type, push_enable=True)
        if ss.check_daily_continuous(coming_next_date=base_bgn_date, calendar=calendar) != 0:
            return -1

        # load and reformat
        fac_data = load_tsdb(
            tsdb_root_dir=self.tsdb_root_dir,
            freq=self.freq,
            value_columns=[self.fac_val_col],
            bgn_date=base_bgn_date,
            end_date=base_end_date,
        )
        ret_data = load_tsdb(
            tsdb_root_dir=self.tsdb_root_dir,
            freq=self.freq,
            value_columns=[self.ret_val_col],
            bgn_date=base_bgn_date,
            end_date=base_end_date,
        )
        rft_fac_data = self.reformat(
            raw_data=fac_data,
            rename_mapper={self.fac_val_col: self.ic_fac.fac_name},
            selected_cols=["tp", "trading_day", "ticker", self.ic_fac.fac_name],
        )
        rft_ret_data = self.reformat(
            raw_data=ret_data,
            rename_mapper={self.ret_val_col: self.ic_ret.ret_name},
            selected_cols=["tp", "trading_day", "ticker", self.ic_ret.ret_name],
        )
        if self.check_fac_and_ret(rft_fac_data, rft_ret_data):
            rft_fac_and_ret_data = self.merge_fac_and_ret(rft_fac_data, rft_ret_data)
            available_data = self.load_available(bgn_date=base_bgn_date, end_date=base_end_date)
            avlb_fac_and_ret_data = self.merge_to_avlb(available_data, rft_fac_and_ret_data)
            ic_data = self.get_ic(avlb_fac_and_ret_data, verbose=verbose)
            ss.append_from_DataFrame(new_data=ic_data.reset_index(), new_data_type=new_data_type)
        return 0


def process_cal_by_fac_ret_test(
    ic_fac: CIcFac,
    ic_ret: CIcRet,
    shift: int,
    tsdb_root_dir: str,
    available_dir: str,
    ic_save_dir: str,
    freq: str,
    sectors: list[str],
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    verbose: bool,
):
    ic_tests = CICTests(
        ic_fac=ic_fac,
        ic_ret=ic_ret,
        shift=shift,
        tsdb_root_dir=tsdb_root_dir,
        available_dir=available_dir,
        ic_save_dir=ic_save_dir,
        freq=freq,
        sectors=sectors,
    )
    ic_tests.main_ic_test(bgn_date=bgn_date, end_date=end_date, calendar=calendar, verbose=verbose)
    return 0


@qtimer
def process_cal_by_factor(
    cfg_factor: CCfgFactor,
    ret_class: TReturnClass,
    ret_names: TReturnNames,
    shift: int,
    tsdb_root_dir: str,
    prefix_user: list[str],
    available_dir: str,
    ic_save_dir: str,
    freq: str,
    sectors: list[str],
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    call_multiprocess: bool,
    processes: int | None,
    verbose: bool,
):
    print(f"[INF] Processing ic calculating for {SFG(str(cfg_factor))} ...")
    combs = cfg_factor.get_combs_neu()  # Test neutralized factors ONLY.
    iter_args: list[tuple[CIcFac, CIcRet]] = []
    for factor_class, factor_names, sub_directory in combs:
        for factor_name, return_name in product(factor_names, ret_names):
            ic_fac = CIcFac(TPrefix(prefix_user + [sub_directory]), fac_class=factor_class, fac_name=factor_name)
            ic_ret = CIcRet(TPrefix(prefix_user + ["Y"]), ret_class=ret_class, ret_name=return_name)
            iter_args.append((ic_fac, ic_ret))

    if call_multiprocess:
        with mp.get_context("spawn").Pool(processes=processes) as pool:
            for ic_fac, ic_ret in iter_args:
                pool.apply_async(
                    process_cal_by_fac_ret_test,
                    kwds={
                        "ic_fac": ic_fac,
                        "ic_ret": ic_ret,
                        "shift": shift,
                        "tsdb_root_dir": tsdb_root_dir,
                        "available_dir": available_dir,
                        "ic_save_dir": ic_save_dir,
                        "freq": freq,
                        "sectors": sectors,
                        "bgn_date": bgn_date,
                        "end_date": end_date,
                        "calendar": calendar,
                        "verbose": verbose,
                    },
                    error_callback=error_handler,
                )
            pool.close()  # necessary for apply_async
            pool.join()  # necessary for apply_async
    else:
        for ic_fac, ic_ret in iter_args:
            process_cal_by_fac_ret_test(
                ic_fac=ic_fac,
                ic_ret=ic_ret,
                shift=shift,
                tsdb_root_dir=tsdb_root_dir,
                available_dir=available_dir,
                ic_save_dir=ic_save_dir,
                freq=freq,
                sectors=sectors,
                bgn_date=bgn_date,
                end_date=end_date,
                calendar=calendar,
                verbose=verbose,
            )
    return 0


@qtimer
def main_cal_ic_tests(
    cfg_factors: list[CCfgFactor],
    ret_class: TReturnClass,
    ret_names: TReturnNames,
    shift: int,
    tsdb_root_dir: str,
    prefix_user: list[str],
    available_dir: str,
    ic_save_dir: str,
    freq: str,
    sectors: list[str],
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    call_multiprocess: bool,
    processes: int | None,
    verbose: bool,
):
    for cfg_factor in track(cfg_factors, description="[INF] Calculating ic-tests for factors"):
        process_cal_by_factor(
            cfg_factor=cfg_factor,
            ret_class=ret_class,
            ret_names=ret_names,
            shift=shift,
            tsdb_root_dir=tsdb_root_dir,
            prefix_user=prefix_user,
            available_dir=available_dir,
            ic_save_dir=ic_save_dir,
            freq=freq,
            sectors=sectors,
            bgn_date=bgn_date,
            end_date=end_date,
            calendar=calendar,
            call_multiprocess=call_multiprocess,
            processes=processes,
            verbose=verbose,
        )
    return 0


"""
Part II: Plot cumulative sum of IC
"""


def process_plt_by_fac_ret_test(
    fac_name: TFactorName,
    ret_name: TReturnName,
    ic_save_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
):
    ss_path = os.path.join(ic_save_dir, ret_name, f"{fac_name}.ss")
    if not os.path.exists(ss_path):
        print(f"[INF] {SFY(ss_path)} does not exist, please check again")
        return 0

    ss = PySharedStackPlus(ss_path)
    df = pd.DataFrame(ss.read_all())
    if df.empty:
        print(f"[INF] {SFY(ss_path)} has no data, please check again")
        return 0

    df = df.drop(axis=1, labels=["tp"])
    df = df.set_index("trading_day").truncate(before=int(bgn_date), after=int(end_date))
    df.index = df.index.map(str)
    cumsum_df = df.cumsum()
    plot_lines(data=cumsum_df, fig_save_dir=os.path.join(ic_save_dir, f"{ret_name}-plot"), fig_name=fac_name)
    return 0


def process_plt_by_factor(
    cfg_factor: CCfgFactor,
    ret_names: TReturnNames,
    ic_save_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    call_multiprocess: bool,
    processes: int | None,
):
    print(f"[INF] Processing plotting ic-cumsum {SFG(str(cfg_factor))} ...")
    combs = cfg_factor.get_combs_neu()
    iter_args: list[tuple[TFactorName, TReturnName]] = []
    for _, factor_names, _ in combs:
        for factor_name, return_name in product(factor_names, ret_names):
            iter_args.append((factor_name, return_name))

    if call_multiprocess:
        with mp.get_context("spawn").Pool(processes=processes) as pool:
            for fac_name, ret_name in iter_args:
                pool.apply_async(
                    process_plt_by_fac_ret_test,
                    kwds={
                        "fac_name": fac_name,
                        "ret_name": ret_name,
                        "ic_save_dir": ic_save_dir,
                        "bgn_date": bgn_date,
                        "end_date": end_date,
                    },
                    error_callback=error_handler,
                )
            pool.close()  # necessary for apply_async
            pool.join()  # necessary for apply_async
    else:
        for fac_name, ret_name in iter_args:
            process_plt_by_fac_ret_test(
                fac_name=fac_name,
                ret_name=ret_name,
                ic_save_dir=ic_save_dir,
                bgn_date=bgn_date,
                end_date=end_date,
            )


@qtimer
def main_plt_ic_tests(
    cfg_factors: list[CCfgFactor],
    ret_names: TReturnNames,
    ic_save_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    call_multiprocess: bool,
    processes: int | None = None,
):
    for cfg_factor in track(cfg_factors, description="[INF] Plotting ic-tests for factors"):
        process_plt_by_factor(
            cfg_factor=cfg_factor,
            ret_names=ret_names,
            ic_save_dir=ic_save_dir,
            bgn_date=bgn_date,
            end_date=end_date,
            call_multiprocess=call_multiprocess,
            processes=processes,
        )
    return 0


"""
Part III: Rolling ICIR
"""


def process_rolling_by_fac_ret_test(
    fac_name: TFactorName,
    ret_name: TReturnName,
    rolling_win: int,  # one of [60, 120, 240]
    shift: int,
    sectors: list[str],
    ic_save_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
):
    ss_path = os.path.join(ic_save_dir, ret_name, fac_name + ".ss")
    if not os.path.exists(ss_path):
        print(f"[INF] {SFY(ss_path)} does not exist, please check again")
        return 0

    ss = PySharedStackPlus(ss_path)
    df = pd.DataFrame(ss.read_all())
    if df.empty:
        print(f"[INF] {SFY(ss_path)} has no data, please check again")
        return 0

    roll_bgn_date = calendar.get_next_date(bgn_date, -shift - rolling_win + 1)
    base_bgn_date = calendar.get_next_date(bgn_date, -shift)
    base_end_date = calendar.get_next_date(end_date, -shift)

    df = df.set_index("trading_day").truncate(before=int(roll_bgn_date), after=int(base_end_date)).reset_index()
    df = df.set_index(["tp", "trading_day"])[sectors]
    df_mu = df.rolling(window=rolling_win).mean()
    df_sd = df.rolling(window=rolling_win).std()
    df_ir = df_mu / df_sd
    df_rolling_ic = df_mu.truncate(before=int(base_bgn_date)).reset_index()
    df_rolling_ir = df_ir.truncate(before=int(base_bgn_date)).reset_index()

    # set new data type
    new_data_type = np.dtype([("tp", np.int64), ("trading_day", np.int32)] + [(s, np.float32) for s in sectors])

    # save rolling ic
    ss_path = os.path.join(ic_save_dir, f"{ret_name}-roll-W{rolling_win:03d}-ic", fac_name + ".ss")
    ss = PySharedStackPlus(ss_path, dtype=new_data_type, push_enable=True)
    if ss.check_daily_continuous(base_bgn_date, calendar) == 0:
        ss.append_from_DataFrame(new_data=df_rolling_ic, new_data_type=new_data_type)

    # save rolling ir
    ss_path = os.path.join(ic_save_dir, f"{ret_name}-roll-W{rolling_win:03d}-ir", fac_name + ".ss")
    ss = PySharedStackPlus(ss_path, dtype=new_data_type, push_enable=True)
    if ss.check_daily_continuous(base_bgn_date, calendar) == 0:
        ss.append_from_DataFrame(new_data=df_rolling_ir, new_data_type=new_data_type)

    return 0


def process_rolling_by_factor(
    cfg_factor: CCfgFactor,
    rolling_win: int,  # one of [60, 120, 240]
    ret_names: TReturnNames,
    shift: int,
    sectors: list[str],
    ic_save_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    call_multiprocess: bool,
    processes: int | None,
):
    print(f"[INF] Processing rolling icir for {SFG(str(cfg_factor))} ...")
    combs = cfg_factor.get_combs_neu()
    iter_args: list[tuple[TFactorName, TReturnName]] = []
    for _, factor_names, _ in combs:
        for factor_name, return_name in product(factor_names, ret_names):
            iter_args.append((factor_name, return_name))

    if call_multiprocess:
        with mp.get_context("spawn").Pool(processes=processes) as pool:
            for fac_name, ret_name in iter_args:
                pool.apply_async(
                    process_rolling_by_fac_ret_test,
                    kwds={
                        "fac_name": fac_name,
                        "ret_name": ret_name,
                        "rolling_win": rolling_win,
                        "shift": shift,
                        "sectors": sectors,
                        "ic_save_dir": ic_save_dir,
                        "bgn_date": bgn_date,
                        "end_date": end_date,
                        "calendar": calendar,
                    },
                    error_callback=error_handler,
                )
            pool.close()  # necessary for apply_async
            pool.join()  # necessary for apply_async
    else:
        for fac_name, ret_name in iter_args:
            process_rolling_by_fac_ret_test(
                fac_name=fac_name,
                ret_name=ret_name,
                rolling_win=rolling_win,
                shift=shift,
                sectors=sectors,
                ic_save_dir=ic_save_dir,
                bgn_date=bgn_date,
                end_date=end_date,
                calendar=calendar,
            )
    return 0


@qtimer
def main_rol_ic_tests(
    cfg_factors: list[CCfgFactor],
    rolling_wins: list[int],  # [60, 120, 240]
    ret_names: TReturnNames,
    shift: int,
    sectors: list[str],
    ic_save_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    call_multiprocess: bool,
    processes: int | None = None,
):
    with Progress() as pb:
        wins_task = pb.add_task(description="[INF] Calculating IC-IR for rolling windows", total=len(rolling_wins))
        facs_task = pb.add_task(description="[INF] Calculating IC-IR for factor class", total=len(cfg_factors))
        for rolling_win in rolling_wins:
            pb.update(facs_task, completed=0)
            for cfg_factor in cfg_factors:
                process_rolling_by_factor(
                    cfg_factor=cfg_factor,
                    rolling_win=rolling_win,
                    ret_names=ret_names,
                    shift=shift,
                    sectors=sectors,
                    ic_save_dir=ic_save_dir,
                    bgn_date=bgn_date,
                    end_date=end_date,
                    calendar=calendar,
                    call_multiprocess=call_multiprocess,
                    processes=processes,
                )
                pb.update(facs_task, advance=1)
            pb.update(wins_task, advance=1)
    return 0


"""
Part IV: Summary by sector
"""


class CSlcFacFromICIR:
    def __init__(
        self,
        ret_class: TReturnClass,
        ret_name: TReturnName,
        shift: int,
        rolling_win: int,
        cfg_factors: list[CCfgFactor],
        sectors: list[str],
        ic_save_dir: str,
    ):
        self.ret_class = ret_class
        self.ret_name = ret_name
        self.shift = shift
        self.rolling_win = rolling_win
        self.cfg_factors = cfg_factors
        self.sectors = sectors
        self.ic_save_dir = ic_save_dir

    @staticmethod
    def data_settings() -> dict:
        return {
            "tp": np.int64,
            "trading_day": np.int32,
            "factor_class": "S40",
            "factor_name": "S40",
            "ic": np.float32,
            "ir": np.float32,
            "rank_ic": np.float32,
            "rank_ir": np.float32,
            "rank": np.float32,
        }

    @property
    def save_data_type(self) -> np.dtype:
        return np.dtype([(k, v) for k, v in self.data_settings().items()])

    @property
    def save_cols(self) -> list[str]:
        return list(self.data_settings())

    def check_sector_continuity(self, sector: str, base_bgn_date: np.int32, calendar: CCalendar) -> bool:
        ss_path = os.path.join(self.ic_save_dir, self.save_id(), sector + ".ss")
        ss = PySharedStackPlus(ss_path, dtype=self.save_data_type, push_enable=True)
        return ss.check_daily_continuous(base_bgn_date, calendar) == 0

    def check_sectors_continuity(self, base_bgn_date: np.int32, calendar: CCalendar):
        res = [self.check_sector_continuity(sector, base_bgn_date, calendar) for sector in self.sectors]
        return all(res)

    def load_range_by_fac(self, fac_name: str, icir_type: str, bgn_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        if icir_type not in ["ic", "ir"]:
            raise ValueError(f"rolling type = {icir_type} is illegal, must be one of ['ic', 'ir']")
        ret_fac_dir = f"{self.ret_name}-roll-W{self.rolling_win:03d}-{icir_type}"
        ss_path = os.path.join(self.ic_save_dir, ret_fac_dir, fac_name + ".ss")
        ss = PySharedStackPlus(ss_path)
        fac_data = pd.DataFrame(ss.read_all())
        fac_data = fac_data.set_index("trading_day").truncate(before=int(bgn_date), after=int(end_date)).reset_index()
        return fac_data

    def load_range(self, base_bgn_date, base_end_date) -> tuple[pd.DataFrame, pd.DataFrame]:
        ic_dfs: list[pd.DataFrame] = []
        ir_dfs: list[pd.DataFrame] = []
        for cfg_factor in self.cfg_factors:
            combs = cfg_factor.get_combs_neu()
            for factor_class, factor_names, _ in combs:
                for fac_name in factor_names:
                    fac_ic_data = self.load_range_by_fac(fac_name, "ic", base_bgn_date, base_end_date)
                    fac_ic_data["factor_class"] = factor_class
                    fac_ic_data["factor_name"] = fac_name
                    ic_dfs.append(fac_ic_data)

                    fac_ir_data = self.load_range_by_fac(fac_name, "ir", base_bgn_date, base_end_date)
                    fac_ir_data["factor_class"] = factor_class
                    fac_ir_data["factor_name"] = fac_name
                    ir_dfs.append(fac_ir_data)
        ic_data = pd.concat(ic_dfs, axis=0, ignore_index=True)
        ir_data = pd.concat(ir_dfs, axis=0, ignore_index=True)
        return ic_data, ir_data

    def check_ic_and_ir(self, ic_data: pd.DataFrame, ir_data: pd.DataFrame, ic_ir_data: pd.DataFrame):
        ic_len = len(ic_data)
        ir_len = len(ir_data)
        ic_ir_len = len(ic_ir_data)
        if ic_len == ir_len == ic_ir_len:
            return True
        else:
            print(f"[{SFR('ERR')}] Data is not aligned of {SFG(self.ret_name)} and {SFG(self.rolling_win)}")
            print(f"[{SFR('ERR')}] Size of ic           = {ic_len}")
            print(f"[{SFR('ERR')}] Size of ir           = {ir_len}")
            print(f"[{SFR('ERR')}] Size of merged ic ir = {ic_ir_len}")
            return False

    def select_factor(
        self, trading_day_ic_ir_data: pd.DataFrame, col_ic: str, col_ir: str, sector: str
    ) -> pd.DataFrame:
        raise NotImplementedError

    def save_id(self) -> str:
        raise NotImplementedError

    def save(self, sector: str, new_data: pd.DataFrame, new_data_type: np.dtype):
        ss_path = os.path.join(self.ic_save_dir, self.save_id(), sector + ".ss")
        ss = PySharedStackPlus(ss_path, dtype=new_data_type, push_enable=True)
        ss.append_from_DataFrame(new_data=new_data, new_data_type=new_data_type)
        return 0

    def process_by_sector(
        self,
        sector: str,
        sector_data: pd.DataFrame,
        groupby_keys: str | list[str],
        col_ic: str,
        col_ir: str,
    ):
        effect_data = sector_data.dropna(axis=0, how="all", subset=[col_ic, col_ir])
        if effect_data.empty:
            raise ValueError(f"There is not data for sector {sector}")
        else:
            selected_factors = effect_data.groupby(by=groupby_keys, group_keys=False).apply(
                self.select_factor,  # type:ignore
                col_ic=col_ic,
                col_ir=col_ir,
                sector=sector,
            )
            selected_factors = selected_factors.rename(mapper={col_ic: "ic", col_ir: "ir"}, axis=1)
            selected_factors = selected_factors[self.save_cols]
            self.save(sector=sector, new_data=selected_factors, new_data_type=self.save_data_type)
            return 0

    @qtimer
    def main_slc_by_ret_and_roll_win(
        self,
        bgn_date: np.int32,
        end_date: np.int32,
        calendar: CCalendar,
        call_multiprocess: bool,
    ):
        # dates and continuity
        base_bgn_date = calendar.get_next_date(bgn_date, -self.shift)
        base_end_date = calendar.get_next_date(end_date, -self.shift)
        if not self.check_sectors_continuity(base_bgn_date, calendar):
            return -1

        # check shape
        ic_data, ir_data = self.load_range(base_bgn_date, base_end_date)
        ic_ir_data = pd.merge(
            left=ic_data,
            right=ir_data,
            on=["tp", "trading_day", "factor_class", "factor_name"],
            how="inner",
            suffixes=("_ic", "_ir"),
        )
        if not self.check_ic_and_ir(ic_data, ir_data, ic_ir_data):
            return -1

        # core
        groupby_keys = ["tp", "trading_day"]
        if call_multiprocess:
            with mp.get_context("spawn").Pool() as pool:
                for sector in self.sectors:
                    col_ic, col_ir = f"{sector}_ic", f"{sector}_ir"
                    sector_data = ic_ir_data[groupby_keys + ["factor_class", "factor_name"] + [col_ic, col_ir]]
                    pool.apply_async(
                        self.process_by_sector,
                        kwds={
                            "sector": sector,
                            "sector_data": sector_data,
                            "groupby_keys": groupby_keys,
                            "col_ic": col_ic,
                            "col_ir": col_ir,
                        },
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
        else:
            for sector in self.sectors:
                col_ic, col_ir = f"{sector}_ic", f"{sector}_ir"
                sector_data = ic_ir_data[groupby_keys + ["factor_class", "factor_name"] + [col_ic, col_ir]]
                self.process_by_sector(
                    sector=sector,
                    sector_data=sector_data,
                    groupby_keys=groupby_keys,
                    col_ic=col_ic,
                    col_ir=col_ir,
                )
        return 0


class CSlcFacFromICIRRank(CSlcFacFromICIR):
    def __init__(
        self,
        top_ratio: float,
        init_threshold: float,
        ret_class: TReturnClass,
        ret_name: TReturnName,
        shift: int,
        rolling_win: int,
        cfg_factors: list[CCfgFactor],
        sectors: list[str],
        ic_save_dir: str,
    ):
        self.top_ratio = top_ratio
        self.init_threshold = init_threshold
        super().__init__(
            ret_class=ret_class,
            ret_name=ret_name,
            shift=shift,
            rolling_win=rolling_win,
            cfg_factors=cfg_factors,
            sectors=sectors,
            ic_save_dir=ic_save_dir,
        )

    def save_id(self) -> str:
        return f"{self.ret_name}-roll-W{self.rolling_win:03d}-SlcFac-TR{int(self.top_ratio*100):02d}"

    def select_factor(
        self, trading_day_ic_ir_data: pd.DataFrame, col_ic: str, col_ir: str, sector: str
    ) -> pd.DataFrame:
        trading_day = trading_day_ic_ir_data["trading_day"].iloc[0]
        trading_day_ic_ir_data["rank_ic"] = trading_day_ic_ir_data[col_ic].rank()
        trading_day_ic_ir_data["rank_ir"] = trading_day_ic_ir_data[col_ir].rank()
        trading_day_ic_ir_data["rank"] = trading_day_ic_ir_data[["rank_ic", "rank_ir"]].mean(axis=1)
        new_data = trading_day_ic_ir_data.sort_values(by="rank", ascending=True)
        neg_data = new_data[new_data[col_ic] < 0].drop_duplicates(subset="factor_class", keep="first")
        pos_data = new_data[new_data[col_ic] > 0].drop_duplicates(subset="factor_class", keep="last")
        concat_data = pd.concat([neg_data, pos_data], axis=0, ignore_index=False)
        no_duplicated_data = concat_data.drop_duplicates(subset="factor_name", keep="first")
        if no_duplicated_data[col_ic].abs().max() > 0:
            t = self.init_threshold
            selected_data = no_duplicated_data[no_duplicated_data[col_ic].abs() >= t]
            while selected_data.empty:
                # print(f"[INF] IC threshold is too high, lower it for {SFG(sector)} @ {SFG(trading_day)}")
                t = t * 0.8
                selected_data = no_duplicated_data[no_duplicated_data[col_ic].abs() >= t]
                if not selected_data.empty:
                    print(f"[INF] {len(selected_data):>2d} factors are selected for {SFG(sector)} @ {SFG(trading_day)}")
        else:
            if trading_day >= 20160101:
                print(f"[INF] All ic data is 0, no factors are selected for {SFG(sector)} @ {SFG(trading_day)}")
            selected_data = pd.DataFrame()
        return selected_data


@qtimer
def main_sum_ic_tests(
    top_ratios: list[float],
    init_threshold: float,
    ret_class: TReturnClass,
    ret_names: TReturnNames,
    shift: int,
    rolling_wins: list[int],  # [60, 120, 240]
    cfg_factors: list[CCfgFactor],
    sectors: list[str],
    ic_save_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    call_multiprocess: bool,
):
    iter_args: list[tuple[TReturnName, int, float]] = list(product(ret_names, rolling_wins, top_ratios))
    for ret_name, rolling_win, top_ratio in track(iter_args, description=f"[INF] Processing selection of factors"):
        print(f"[INF] Processing with {SFG(ret_name)}, rolling window = {SFG(rolling_win)}, ratio = {SFG(top_ratio)}")
        selected_factors = CSlcFacFromICIRRank(
            top_ratio=top_ratio,
            init_threshold=init_threshold,
            ret_class=ret_class,
            ret_name=ret_name,
            shift=shift,
            rolling_win=rolling_win,
            cfg_factors=cfg_factors,
            sectors=sectors,
            ic_save_dir=ic_save_dir,
        )
        selected_factors.main_slc_by_ret_and_roll_win(
            bgn_date=bgn_date,
            end_date=end_date,
            calendar=calendar,
            call_multiprocess=call_multiprocess,
        )
    return 0


"""
Part V: SlcFacReader
"""


class CSlcFacReader:
    def __init__(
        self,
        sector: str,
        ret_class: TReturnClass,
        ret_name: TReturnName,
        shift: int,
        top_ratio: float,
        rolling_win: int,  # one of [60, 120, 240]
        ic_save_dir: str,
    ):
        self.sector = sector
        self.ret_class, self.ret_name, self.shift = ret_class, ret_name, shift
        self.top_ratio = top_ratio
        self.rolling_win = rolling_win
        self.ic_save_dir = ic_save_dir

        self.slc_fac_data: pd.DataFrame = self.load()

    def save_id(self) -> str:
        return f"{self.ret_name}-roll-W{self.rolling_win:03d}-SlcFac-TR{int(self.top_ratio*100):02d}"

    def load(self) -> pd.DataFrame:
        ss_path = os.path.join(self.ic_save_dir, self.save_id(), self.sector + ".ss")
        ss = PySharedStackPlus(ss_path)
        slc_fac_data = pd.DataFrame(ss.read_all()).set_index("trading_day")
        return slc_fac_data

    def get_slc_facs(self, trading_day: np.int32) -> list[tuple[str, str]]:
        trading_day_data = self.slc_fac_data.loc[trading_day]
        res = []
        if isinstance(trading_day_data, pd.Series):
            factor_class, factor_name = trading_day_data["factor_class"], trading_day_data["factor_name"]
            res.append((factor_class.decode("utf-8"), factor_name.decode("utf-8")))
        elif isinstance(trading_day_data, pd.DataFrame):
            for factor_class, factor_name in zip(trading_day_data["factor_class"], trading_day_data["factor_name"]):
                res.append((factor_class.decode("utf-8"), factor_name.decode("utf-8")))
        else:
            raise TypeError(f"type of trading_day_data is {type(trading_day_data)}")
        return res
