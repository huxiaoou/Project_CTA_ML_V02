import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from rich.progress import track, Progress
from hUtils.tools import qtimer, error_handler
from hUtils.calendar import CCalendar
from hUtils.ioPlus import load_tsdb, PySharedStackPlus
from hSolutions.sMclrn import CTest
from dataclasses import dataclass


@dataclass(frozen=True)
class CSimSig:
    prefix: list[str]
    sid: str

    @property
    def tsdb_val_col(self) -> str:
        return ".".join(self.prefix + [self.sid])


@dataclass(frozen=True)
class CSimRet:
    prefix: list[str]
    ret: str
    shift: int

    @property
    def tsdb_val_col(self) -> str:
        return ".".join(self.prefix + [self.ret])

    @property
    def win(self) -> int:
        return int(self.prefix[-1].split("L")[0])

    @property
    def lag(self) -> int:
        return int(self.prefix[-1].split("L")[1])


@dataclass(frozen=True)
class CSimArg:
    sim_id: str
    sig: CSimSig
    ret: CSimRet
    cost: float


def get_sim_args_from_tests(tests: list[CTest], prefix_user: list[str], cost: float) -> list[CSimArg]:
    res: list[CSimArg] = []
    for test in tests:
        sig = CSimSig(prefix=prefix_user + ["signals"] + test.prefix, sid=test.ret.ret_name)
        if test.ret.ret_name.startswith("Open"):
            ret_class, ret_name = "001L1", "OpenRtn001L1"
            # ret_class, ret_name = "010L1", "OpenRtn010L1"
        elif test.ret.ret_name.startswith("Close"):
            ret_class, ret_name = "001L1", "CloseRtn001L1"
            # ret_class, ret_name = "010L1", "CloseRtn010L1"
        else:
            raise ValueError(f"ret_name = {test.ret.ret_name} is illegal")
        ret = CSimRet(prefix=prefix_user + ["Y"] + [ret_class], ret=ret_name, shift=test.ret.shift)
        sim_id = ".".join(test.prefix + [test.ret.ret_name])
        sim_arg = CSimArg(sim_id=sim_id, sig=sig, ret=ret, cost=cost)
        res.append(sim_arg)
    return res


def group_sim_args(sim_args: list[CSimArg]) -> dict[tuple[str, str, str, str], dict[tuple[str, str], CSimArg]]:
    grouped_sim_args: dict[tuple[str, str, str, str], dict[tuple[str, str], CSimArg]] = {}
    for sim_arg in sim_args:
        ret_class, trn_win, model_desc, sector, unique_id, ret_name = sim_arg.sim_id.split(".")
        key0, key1 = (ret_class, trn_win, model_desc, ret_name), (sector, unique_id)
        if key0 not in grouped_sim_args:
            grouped_sim_args[key0] = {}
        grouped_sim_args[key0][key1] = sim_arg
    return grouped_sim_args


def group_sim_args_by_sector(sim_args: list[CSimArg]) -> dict[str, dict[tuple[str, str, str, str, str], CSimArg]]:
    grouped_sim_args: dict[str, dict[tuple[str, str, str, str, str], CSimArg]] = {}
    for sim_arg in sim_args:
        ret_class, trn_win, model_desc, sector, unique_id, ret_name = sim_arg.sim_id.split(".")
        key0, key1 = sector, (ret_class, trn_win, model_desc, ret_name, unique_id)
        if key0 not in grouped_sim_args:
            grouped_sim_args[key0] = {}
        grouped_sim_args[key0][key1] = sim_arg
    return grouped_sim_args


@dataclass(frozen=True)
class CPortfolioArg:
    pid: str
    target: str
    weights: dict[str, float]
    portfolio_sim_args: dict[str, CSimArg]


def get_portfolio_args(portfolios: dict[str, dict], sim_args: list[CSimArg]) -> list[CPortfolioArg]:
    res: list[CPortfolioArg] = []
    for portfolio_id, portfolio_cfg in portfolios.items():
        target, weights = portfolio_cfg["target"], portfolio_cfg["weights"]
        portfolio_sim_args = {}
        for sim_arg in sim_args:
            *_, unique_id, ret_name = sim_arg.sim_id.split(".")
            if (unique_id in weights) and (ret_name == target):
                portfolio_sim_args[unique_id] = sim_arg
        portfolio_arg = CPortfolioArg(portfolio_id, target, weights, portfolio_sim_args)
        res.append(portfolio_arg)
    return res


def get_sim_args_from_portfolios(
    portfolios: dict[str, dict], prefix_user: list[str], cost: float, shift: int
) -> list[CSimArg]:
    res: list[CSimArg] = []
    for portfolio_id, portfolio_cfg in portfolios.items():
        target: str = portfolio_cfg["target"]
        sig = CSimSig(prefix=prefix_user + ["signals", "portfolios", portfolio_id], sid=target)
        if target.startswith("Open"):
            ret_class, ret_name = "001L1", "OpenRtn001L1"
            # ret_class, ret_name = "010L1", "OpenRtn010L1"
        elif target.startswith("Close"):
            ret_class, ret_name = "001L1", "CloseRtn001L1"
            # ret_class, ret_name = "010L1", "CloseRtn010L1"
        else:
            raise ValueError(f"ret_name = {target} is illegal")
        ret = CSimRet(prefix=prefix_user + ["Y"] + [ret_class], ret=ret_name, shift=shift)
        sim_arg = CSimArg(sim_id=portfolio_id, sig=sig, ret=ret, cost=cost)
        res.append(sim_arg)
    return res


class CSim:
    def __init__(self, sim_arg: CSimArg, tsdb_root_dir: str, freq: str, sim_save_dir: str):
        self.sim_id = sim_arg.sim_id
        self.sig = sim_arg.sig
        self.ret = sim_arg.ret
        self.cost = sim_arg.cost
        self.tsdb_root_dir = tsdb_root_dir
        self.freq = freq
        self.sim_save_dir = sim_save_dir

    @property
    def data_settings(self) -> dict:
        return {
            "tp": np.int64,
            "trading_day": np.int32,
            "raw_ret": np.float64,
            "dlt_wgt": np.float64,
            "cost": np.float64,
            "net_ret": np.float64,
            "nav": np.float64,
        }

    @property
    def data_types(self) -> np.dtype:
        return np.dtype([(k, v) for k, v in self.data_settings.items()])

    @property
    def sim_save_path(self) -> str:
        return os.path.join(self.sim_save_dir, f"{self.sim_id}.ss")

    def check_continuous(self, bgn_date: np.int32, calendar: CCalendar) -> bool:
        ss = PySharedStackPlus(self.sim_save_path, self.data_types, push_enable=True)
        return ss.check_daily_continuous(bgn_date, calendar) == 0

    @property
    def nav_empty(self) -> bool:
        ss = PySharedStackPlus(self.sim_save_path, self.data_types)
        return ss.size() == 0

    @property
    def last_nav(self) -> np.float64:
        ss = PySharedStackPlus(self.sim_save_path, self.data_types)
        if ss.size() > 0:
            return ss.last_nav
        else:
            return np.float64(1.0)

    def reformat_sig(self, sig_data: pd.DataFrame) -> pd.DataFrame:
        new_data = sig_data.rename(mapper={self.sig.tsdb_val_col: "sig"}, axis=1)
        new_data = new_data[["tp", "trading_day", "ticker", "sig"]].fillna(0)
        return new_data

    def reformat_ret(self, ret_data: pd.DataFrame) -> pd.DataFrame:
        new_data = ret_data.rename(mapper={self.ret.tsdb_val_col: "ret"}, axis=1)
        new_data = new_data[["tp", "trading_day", "ticker", "ret"]].fillna(0)
        new_data["ret"] = new_data["ret"] / self.ret.win
        return new_data

    def merge_sig_and_ret(self, sig_data: pd.DataFrame, ret_data: pd.DataFrame) -> pd.DataFrame:
        merged_data = pd.merge(left=sig_data, right=ret_data, on=["tp", "trading_day", "ticker"], how="inner")
        return merged_data.dropna(axis=0, subset=["sig"], how="any")

    def cal_ret(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        raw_ret = merged_data.groupby(by=["tp", "trading_day"], group_keys=True).apply(lambda z: z["sig"] @ z["ret"])
        wgt_data = pd.pivot_table(
            data=merged_data,
            index=["tp", "trading_day"],
            columns=["ticker"],
            values="sig",
            aggfunc="mean",
        ).fillna(0)
        wgt_data_prev = wgt_data.shift(1).fillna(0)
        wgt_diff = wgt_data - wgt_data_prev
        dlt_wgt = wgt_diff.abs().sum(axis=1)
        cost = dlt_wgt * self.cost
        net_ret = raw_ret - cost
        sim_data = pd.DataFrame({"raw_ret": raw_ret, "dlt_wgt": dlt_wgt, "cost": cost, "net_ret": net_ret})
        return sim_data

    def align_dates(self, sim_data: pd.DataFrame, bgn_date: np.int32, calendar: CCalendar) -> pd.DataFrame:
        aligned_sim_data = sim_data.reset_index()
        aligned_sim_data["trading_day"] = aligned_sim_data["trading_day"].map(
            lambda z: calendar.get_next_date(z, self.ret.shift)
        )
        aligned_sim_data["tp"] = aligned_sim_data["trading_day"].map(
            lambda z: calendar.convert_date_to_tp(z, "16:00:00")
        )
        return aligned_sim_data[aligned_sim_data["trading_day"] >= bgn_date]

    def update_nav(self, aligned_sim_data: pd.DataFrame, last_nav: np.float64) -> pd.DataFrame:
        aligned_sim_data["nav"] = (aligned_sim_data["net_ret"] + 1).cumprod() * last_nav  # type:ignore
        return aligned_sim_data

    def save(self, new_data: pd.DataFrame):
        ss = PySharedStackPlus(self.sim_save_path, self.data_types, push_enable=True)
        ss.append_from_DataFrame(new_data, new_data_type=self.data_types)
        return 0

    def main(self, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar):
        if self.check_continuous(bgn_date, calendar):
            d = 0 if self.nav_empty else 1
            base_bgn_date = calendar.get_next_date(bgn_date, -self.ret.shift - d)
            base_end_date = calendar.get_next_date(end_date, -self.ret.shift)
            sig_data = load_tsdb(
                tsdb_root_dir=self.tsdb_root_dir,
                freq=self.freq,
                value_columns=[self.sig.tsdb_val_col],
                bgn_date=base_bgn_date,
                end_date=base_end_date,
            )
            ret_data = load_tsdb(
                tsdb_root_dir=self.tsdb_root_dir,
                freq=self.freq,
                value_columns=[self.ret.tsdb_val_col],
                bgn_date=base_bgn_date,
                end_date=base_end_date,
            )
            rft_sig_data, rft_ret_data = self.reformat_sig(sig_data), self.reformat_ret(ret_data)
            merged_data = self.merge_sig_and_ret(sig_data=rft_sig_data, ret_data=rft_ret_data)
            sim_data = self.cal_ret(merged_data=merged_data)
            aligned_sim_data = self.align_dates(sim_data, bgn_date=bgn_date, calendar=calendar)
            aligned_sim_data[["raw_ret", "net_ret"]] = aligned_sim_data[["raw_ret", "net_ret"]].fillna(0)
            new_data = self.update_nav(aligned_sim_data=aligned_sim_data, last_nav=self.last_nav)
            null_data = new_data[new_data.isnull().any(axis=1)]
            if not null_data.empty:
                raise ValueError(f"{self.sim_id} has nan data")
            self.save(new_data=new_data)
        return 0


def process_for_sim(
    sim_arg: CSimArg,
    tsdb_root_dir: str,
    freq: str,
    sim_save_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
):
    sim = CSim(sim_arg=sim_arg, tsdb_root_dir=tsdb_root_dir, freq=freq, sim_save_dir=sim_save_dir)
    sim.main(bgn_date, end_date, calendar)
    return 0


@qtimer
def main_simulations(
    sim_args: list[CSimArg],
    tsdb_root_dir: str,
    freq: str,
    sim_save_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    call_multiprocess: bool,
):
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description="[INF] Calculating simulations", total=len(sim_args))
            with mp.get_context("spawn").Pool() as pool:
                for sim_arg in sim_args:
                    pool.apply_async(
                        process_for_sim,
                        kwds={
                            "sim_arg": sim_arg,
                            "tsdb_root_dir": tsdb_root_dir,
                            "freq": freq,
                            "sim_save_dir": sim_save_dir,
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
        for sim_arg in track(sim_args, description="[INF] Calculating simulations"):
            # pd.set_option("display.max_rows", 200)
            # for sim_arg in sim_args:
            process_for_sim(
                sim_arg=sim_arg,
                tsdb_root_dir=tsdb_root_dir,
                freq=freq,
                sim_save_dir=sim_save_dir,
                bgn_date=bgn_date,
                end_date=end_date,
                calendar=calendar,
            )
    return 0
