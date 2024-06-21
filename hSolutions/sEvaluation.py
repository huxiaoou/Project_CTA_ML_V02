import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from rich.progress import track, Progress
from hUtils.tools import qtimer, error_handler, check_and_mkdir, SFG
from hSolutions.sSimulation import CSimArg, group_sim_args, group_sim_args_by_sector
from hUtils.nav import CNAV
from hUtils.ioPlus import PySharedStack
from hUtils.plot import plot_lines

"""
Part I: Basic class
"""


class CEvl:
    def __init__(self, nav_path: str):
        self.ss_path = nav_path

    def load(self) -> pd.DataFrame:
        ss = PySharedStack(self.ss_path)
        return pd.DataFrame(ss.read_all())

    @staticmethod
    def truncate(data: pd.DataFrame, bgn_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        return data.set_index("trading_day").truncate(int(bgn_date), int(end_date)).reset_index()

    def add_arguments(self, d: dict):
        raise NotImplementedError

    def get_ret(self, bgn_date: np.int32, end_date: np.int32) -> pd.Series:
        """
        return : a pd.Series, with string index
        """
        raw_data = self.load()
        nav_data = self.truncate(raw_data, bgn_date, end_date)
        nav_data["trade_date"] = nav_data["trading_day"].map(lambda _: str(_))
        ret_srs = nav_data.set_index("trade_date")["net_ret"]
        return ret_srs

    def main(self, bgn_date: np.int32, end_date: np.int32) -> dict:
        indicators = ("hpr", "retMean", "retStd", "retAnnual", "volAnnual", "sharpeRatio", "calmarRatio", "mdd")
        ret_srs = self.get_ret(bgn_date, end_date)
        nav = CNAV(ret_srs, input_type="RET")
        nav.cal_all_indicators()
        res = nav.to_dict()
        res = {k: v for k, v in res.items() if k in indicators}
        self.add_arguments(res)
        return res


class CEvlTest(CEvl):
    def __init__(self, sim_arg: CSimArg, simulations_dir: str):
        self.sim_arg = sim_arg
        nav_path = os.path.join(simulations_dir, f"{sim_arg.sim_id}.ss")
        super().__init__(nav_path)

    def add_arguments(self, d: dict):
        ret_class, trn_win, model_desc, sector, unique_id, ret_name = self.sim_arg.sim_id.split(".")
        other_arguments = {
            "ret_class": ret_class,
            "trn_win": trn_win,
            "model_desc": model_desc,
            "sector": sector,
            "unique_id": unique_id,
            "ret_name": ret_name,
        }
        d.update(other_arguments)
        return 0


def process_for_evl_test(sim_arg: CSimArg, simulations_dir: str, bgn_date: np.int32, end_date: np.int32) -> dict:
    s = CEvlTest(sim_arg, simulations_dir)
    return s.main(bgn_date, end_date)


@qtimer
def main_eval_tests(
    sim_args: list[CSimArg],
    simulations_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    call_multiprocess: bool,
):
    evl_sims: list[dict] = []
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description="[INF] Calculating evaluations", total=len(sim_args))
            with mp.get_context("spawn").Pool() as pool:
                jobs = []
                for sim_arg in sim_args:
                    job = pool.apply_async(
                        process_for_evl_test,
                        args=(sim_arg, simulations_dir, bgn_date, end_date),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                    jobs.append(job)
                pool.close()
                pool.join()
            evl_sims = [job.get() for job in jobs]
    else:
        for sim_arg in track(sim_args, description=f"[INF] Evaluating simulations"):
            evl = process_for_evl_test(sim_arg, simulations_dir, bgn_date, end_date)
            evl_sims.append(evl)

    evl_data = pd.DataFrame(evl_sims)
    evl_data = evl_data.sort_values(by="sharpeRatio", ascending=False)
    evl_data.insert(0, "calmar", evl_data.pop("calmarRatio"))
    evl_data.insert(0, "sharpe", evl_data.pop("sharpeRatio"))
    evl_data.insert(0, "unique_id", evl_data.pop("unique_id"))

    pd.set_option("display.max_rows", 40)
    pd.set_option("display.float_format", lambda z: f"{z:.3f}")
    for (ret_name, sector), ret_name_data in evl_data.groupby(by=["ret_name", "sector"]):
        if sector == "AGR":
            print("\n")
        print("-" * 180)
        print(f"[INF] {SFG(ret_name)}-{SFG(sector)} models with Best sharpe")  # type:ignore
        print(ret_name_data.head(3))

    for (ret_name, sector), ret_name_data in evl_data.groupby(by=["ret_name", "sector"]):
        if sector == "AGR":
            print("\n")

        uid = ret_name_data["unique_id"].iloc[0]
        sharpe, calmar = ret_name_data["sharpe"].iloc[0], ret_name_data["calmar"].iloc[0]
        win, mdl_desc = ret_name_data["trn_win"].iloc[0], ret_name_data["model_desc"].iloc[0]
        w = {
            "AGR": 3,
            "BLK": 5,
            "CHM": 5,
            "EQT": 3,
            "GLD": 1,
            "MTL": 3,
            "OIL": 3,
        }[sector]
        print(f"{uid}: {w} # {sector} Sharpe = {sharpe:.3f}, Calmar = {calmar:.3f}, {win}, {mdl_desc}")
    return 0


"""
Part II: Plot-Single
"""


def process_for_plot(
    group_key: tuple[str, str, str, str, str],
    sub_grouped_sim_args: dict[tuple[str, str], CSimArg],
    simulations_dir: str,
    fig_save_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
):
    ret_data: dict[str, pd.Series] = {}
    for (sector, unique_id), sim_arg in sub_grouped_sim_args.items():
        s = CEvlTest(sim_arg, simulations_dir)
        ret_data[f"{sector}-{unique_id}"] = s.get_ret(bgn_date, end_date)
    ret_df = pd.DataFrame(ret_data).fillna(0)
    nav_df = (ret_df + 1).cumprod()
    fig_name = "-".join(group_key)

    plot_lines(
        data=nav_df,
        figsize=(16, 9),
        line_width=2,
        colormap="jet",
        fig_name=fig_name,
        fig_save_type="jpg",
        fig_save_dir=fig_save_dir,
    )
    return 0


@qtimer
def main_plot_sims(
    sim_args: list[CSimArg],
    simulations_dir: str,
    evaluations_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    call_multiprocess: bool,
):
    check_and_mkdir(fig_save_dir := os.path.join(evaluations_dir, "plot"))
    grouped_sim_args = group_sim_args(sim_args=sim_args)
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description="[INF] Plotting nav ...", total=len(grouped_sim_args))
            with mp.get_context("spawn").Pool() as pool:
                for group_key, sub_grouped_sim_args in grouped_sim_args.items():
                    pool.apply_async(
                        process_for_plot,
                        args=(group_key, sub_grouped_sim_args, simulations_dir, fig_save_dir, bgn_date, end_date),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for group_key, sub_grouped_sim_args in track(grouped_sim_args.items(), description="[INF] Plotting nav ..."):
            process_for_plot(group_key, sub_grouped_sim_args, simulations_dir, fig_save_dir, bgn_date, end_date)
    return 0


def process_for_plot_by_sector(
    sector: str,
    sector_sim_args: dict[tuple[str, str, str, str, str, str], CSimArg],
    simulations_dir: str,
    fig_save_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    top: int = 10,
):
    ret_data: dict[str, pd.Series] = {}
    for sub_key, sim_arg in sector_sim_args.items():
        sub_id = ".".join(sub_key)
        s = CEvlTest(sim_arg, simulations_dir)
        ret_data[sub_id] = s.get_ret(bgn_date, end_date)
    ret_df = pd.DataFrame(ret_data).fillna(0)

    # selected top sharpe ratio for sector
    mu = ret_df.mean()
    sd = ret_df.std()
    sharpe = mu / sd * np.sqrt(250)
    selected_sub_ids = sharpe.sort_values(ascending=False).index[0:top]
    selected_ret_df = ret_df[selected_sub_ids]

    nav_df = (selected_ret_df + 1).cumprod()
    plot_lines(
        data=nav_df,
        figsize=(16, 9),
        line_width=2,
        colormap="jet",
        fig_name=sector,
        fig_save_type="jpg",
        fig_save_dir=fig_save_dir,
    )
    return 0


@qtimer
def main_plot_sims_by_sector(
    sim_args: list[CSimArg],
    simulations_dir: str,
    evaluations_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    call_multiprocess: bool,
):
    check_and_mkdir(fig_save_dir := os.path.join(evaluations_dir, "plot-by-sector"))
    grouped_sim_args = group_sim_args_by_sector(sim_args=sim_args)
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description="[INF] Plotting nav ...", total=len(grouped_sim_args))
            with mp.get_context("spawn").Pool() as pool:
                for sector, sector_sim_args in grouped_sim_args.items():
                    pool.apply_async(
                        process_for_plot_by_sector,
                        args=(sector, sector_sim_args, simulations_dir, fig_save_dir, bgn_date, end_date),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for sector, sector_sim_args in track(grouped_sim_args.items(), description="[INF] Plotting nav ..."):
            process_for_plot_by_sector(sector, sector_sim_args, simulations_dir, fig_save_dir, bgn_date, end_date)
    return 0


"""
Part III: Evaluate and plot portfolios
"""


class CEvlPortfolio(CEvl):
    def __init__(self, portfolio_id: str, simulations_dir: str):
        self.portfolio_id = portfolio_id
        nav_path = os.path.join(simulations_dir, f"{portfolio_id}.ss")
        super().__init__(nav_path)

    def add_arguments(self, d: dict):
        other_arguments = {"portfolioId": self.portfolio_id}
        d.update(other_arguments)
        return 0


def process_for_evl_portfolio(portfolio_id: str, simulations_dir: str, bgn_date: np.int32, end_date: np.int32) -> dict:
    s = CEvlPortfolio(portfolio_id, simulations_dir)
    return s.main(bgn_date, end_date)


@qtimer
def main_eval_portfolios(
    portfolios: dict[str, dict],
    simulations_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
    call_multiprocess: bool,
):
    evl_sims: list[dict] = []
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description="[INF] Calculating evaluations", total=len(portfolios))
            with mp.get_context("spawn").Pool() as pool:
                jobs = []
                for portfolio_id in portfolios:
                    job = pool.apply_async(
                        process_for_evl_portfolio,
                        args=(portfolio_id, simulations_dir, bgn_date, end_date),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                    jobs.append(job)
                pool.close()
                pool.join()
            evl_sims = [job.get() for job in jobs]
    else:
        for portfolio_id in track(portfolios, description=f"[INF] Evaluating simulations"):
            evl = process_for_evl_portfolio(portfolio_id, simulations_dir, bgn_date, end_date)
            evl_sims.append(evl)

    evl_data = pd.DataFrame(evl_sims)
    evl_data = evl_data.sort_values(by="sharpeRatio", ascending=False)
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.float_format", lambda z: f"{z:.3f}")
    print("[INF] Portfolios performance")
    print(evl_data)
    return 0


@qtimer
def main_plot_portfolios(
    portfolios: dict[str, dict],
    simulations_dir: str,
    evaluations_dir: str,
    bgn_date: np.int32,
    end_date: np.int32,
):
    check_and_mkdir(fig_save_dir := os.path.join(evaluations_dir, "plot-by-portfolio"))
    ret = {}
    for portfolio_id in track(portfolios, description=f"[INF] Plot simulations"):
        s = CEvlPortfolio(portfolio_id, simulations_dir)
        ret[portfolio_id] = s.get_ret(bgn_date, end_date)
    ret_df = pd.DataFrame(ret).fillna(0)
    nav_df = (ret_df + 1).cumprod()

    plot_lines(
        data=nav_df,
        figsize=(16, 9),
        line_width=2,
        colormap="jet",
        fig_name="portfolios",
        fig_save_type="jpg",
        fig_save_dir=fig_save_dir,
    )
    return 0
