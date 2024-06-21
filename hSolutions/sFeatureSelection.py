import os
import datetime as dt
import multiprocessing as mp
import numpy as np
import pandas as pd
from itertools import product
from rich.progress import track, Progress
from hUtils.tools import qtimer, SFG, SFY, SFR, error_handler
from hUtils.instruments import parse_instrument_from_contract
from hUtils.calendar import CCalendar
from hUtils.ioPlus import PySharedStackPlus, load_tsdb
from hUtils.typeDef import CTestFtSlc, TFactor, TFactorClass, TFactorName, TReturnName, CRet


"""
Part II: Class for FeatureSelection
"""


class CFeatureSelection:
    XY_INDEX = ["tp", "trading_day", "ticker", "instru"]
    RANDOM_STATE = 0

    def __init__(
        self,
        test: CTestFtSlc,
        tsdb_root_dir: str,
        tsdb_user_prefix: list[str],
        freq: str,
        avlb_path: str,
        feature_selection_dir: str,
        universe: dict[str, dict[str, str]],
    ):
        self.prototype = NotImplemented
        self.fitted_estimator = NotImplemented

        self.test = test

        self.tsdb_root_dir = tsdb_root_dir
        self.tsdb_user_prefix = tsdb_user_prefix
        self.freq = freq

        self.avlb_path = avlb_path
        self.feature_selection_dir = feature_selection_dir
        self.universe = universe
        self.__x_cols: list[TFactorName] = [n for _, n in self.test.facs_pool]
        self.mapper_name_to_class: dict[TFactorName, TFactorClass] = {n: c for c, n in self.test.facs_pool}

    @property
    def x_cols(self) -> list[TFactorName]:
        return self.__x_cols

    @property
    def y_col(self) -> TReturnName:
        return self.test.ret.ret_name

    @staticmethod
    def data_settings() -> dict:
        return {
            "tp": np.int64,
            "trading_day": np.int32,
            "factor_class": "S40",
            "factor_name": "S40",
        }

    @property
    def save_data_type(self) -> np.dtype:
        return np.dtype([(k, v) for k, v in self.data_settings().items()])

    def load_x(self, bgn_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        usr_prefix = ".".join(self.tsdb_user_prefix)
        value_columns, rename_mapper = [], {}
        for factor_class, factor_name in self.test.facs_pool:
            sub_directory = "neutral_by_instru" if factor_class.endswith("-NEU") else "factors_by_instru"
            tsdb_value_col = f"{usr_prefix}.{sub_directory}.{factor_class}.{factor_name}"
            value_columns.append(tsdb_value_col)
            rename_mapper[tsdb_value_col] = factor_name
        x_data = load_tsdb(
            tsdb_root_dir=self.tsdb_root_dir,
            freq=self.freq,
            value_columns=value_columns,
            bgn_date=bgn_date,
            end_date=end_date,
        ).rename(mapper=rename_mapper, axis=1)
        return x_data[["tp", "trading_day", "ticker"] + self.x_cols]

    def load_y(self, bgn_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        usr_prefix = ".".join(self.tsdb_user_prefix)
        tsdb_value_col = f"{usr_prefix}.Y.{self.test.ret.ret_class}.{self.test.ret.ret_name}"
        value_columns = [tsdb_value_col]
        rename_mapper = {tsdb_value_col: self.test.ret.ret_name}
        y_data = load_tsdb(
            tsdb_root_dir=self.tsdb_root_dir,
            freq=self.freq,
            value_columns=value_columns,
            bgn_date=bgn_date,
            end_date=end_date,
        ).rename(mapper=rename_mapper, axis=1)
        return y_data[["tp", "trading_day", "ticker"] + [self.y_col]]

    def load_sector_available(self) -> pd.DataFrame:
        ss = PySharedStackPlus(self.avlb_path)
        avlb_data = pd.DataFrame(ss.read_all())
        avlb_data["instru"] = avlb_data["instrument"].map(lambda z: z.decode("utf-8"))
        avlb_data["sector"] = avlb_data["sectorL1"].map(lambda z: z.decode("utf-8"))
        filter_sector = avlb_data["sector"] == self.test.sector
        return avlb_data.loc[filter_sector, ["tp", "trading_day", "instru"]]

    def truncate_avlb(self, avlb_data: pd.DataFrame, bgn_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        filter_dates = (avlb_data["trading_day"] >= bgn_date) & (avlb_data["trading_day"] <= end_date)
        return avlb_data[filter_dates]

    def filter_by_sector(self, data: pd.DataFrame, sector_avlb_data: pd.DataFrame) -> pd.DataFrame:
        data["instru"] = data["ticker"].map(lambda z: parse_instrument_from_contract(z.decode("utf-8")))
        new_data = pd.merge(left=sector_avlb_data, right=data, on=["tp", "trading_day", "instru"], how="inner")
        return new_data

    def set_index(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.set_index(self.XY_INDEX)

    def aligned_xy(self, x_data: pd.DataFrame, y_data: pd.DataFrame) -> pd.DataFrame:
        aligned_data = pd.merge(left=x_data, right=y_data, left_index=True, right_index=True, how="inner")
        s0, s1, s2 = len(x_data), len(y_data), len(aligned_data)
        if s0 == s1 == s2:
            return aligned_data
        else:
            print(f"[{SFR('ERR')}] Length of X             = {SFY(s0)}")
            print(f"[{SFR('ERR')}] Length of X             = {SFY(s0)}")
            print(f"[{SFR('ERR')}] Length of y             = {SFY(s1)}")
            print(f"[{SFR('ERR')}] Length of aligned (X,y) = {SFY(s2)}")
            raise ValueError("(X,y) have different lengths")

    def drop_and_fill_nan(self, aligned_data: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
        idx_null = aligned_data.isnull()
        nan_data = aligned_data[idx_null.any(axis=1)]
        if not nan_data.empty:
            # keep rows where nan prop is <= threshold
            filter_nan = (idx_null.sum(axis=1) / aligned_data.shape[1]) <= threshold
            return aligned_data[filter_nan].fillna(0)
        return aligned_data

    def get_X_y(self, aligned_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return aligned_data[self.x_cols], aligned_data[self.y_col]

    def core(self, x_data: pd.DataFrame, y_data: pd.Series) -> list[TFactorName]:
        X, y = x_data.values, y_data.values
        self.fitted_estimator = self.prototype.fit(X, y)
        return []

    def get_factor_class(self, factor_names: list[TFactorName]) -> list[TFactorClass]:
        return [self.mapper_name_to_class[n] for n in factor_names]

    @staticmethod
    def get_selected_data(
        trading_day: np.int32, factor_class: list[TFactorClass], factor_names: list[TFactorName]
    ) -> pd.DataFrame:
        tp = dt.datetime.strptime(f"{trading_day} 16:00:00", "%Y%m%d %H:%M:%S").timestamp() * 10**9
        selected_data = pd.DataFrame(
            {
                "tp": tp,
                "trading_day": trading_day,
                "factor_class": factor_class,
                "factor_names": factor_names,
            }
        )
        return selected_data

    def select(
        self, model_update_day: np.int32, sec_avlb_data: pd.DataFrame, calendar: CCalendar, verbose: bool
    ) -> pd.DataFrame:
        model_update_month = model_update_day // 100
        trn_b_date = calendar.get_next_date(model_update_day, shift=-self.test.ret.shift - self.test.trn_win + 1)
        trn_e_date = calendar.get_next_date(model_update_day, shift=-self.test.ret.shift)
        sec_avlb_data = self.truncate_avlb(sec_avlb_data, trn_b_date, trn_e_date)
        x_data, y_data = self.load_x(trn_b_date, trn_e_date), self.load_y(trn_b_date, trn_e_date)
        x_data, y_data = self.filter_by_sector(x_data, sec_avlb_data), self.filter_by_sector(y_data, sec_avlb_data)
        x_data, y_data = self.set_index(x_data), self.set_index(y_data)
        aligned_data = self.aligned_xy(x_data, y_data)
        aligned_data = self.drop_and_fill_nan(aligned_data)
        X, y = self.get_X_y(aligned_data=aligned_data)
        factor_names = self.core(x_data=X, y_data=y)
        factor_class = self.get_factor_class(factor_names=factor_names)
        selected_data = self.get_selected_data(trn_e_date, factor_class=factor_class, factor_names=factor_names)
        if verbose:
            print(
                f"[INF] Feature selection @ {SFG(int(model_update_day))}, "
                f"factor selected @ {SFG(int(trn_e_date))}, "
                f"using train data @ [{SFG(int(trn_b_date))},{SFG(int(trn_e_date))}], "
                f"save as {SFG(int(model_update_month))}"
            )
        return selected_data

    def save_id(self) -> str:
        return f"{self.test.ret.ret_name}-roll-W{self.test.trn_win:03d}-SlcFac"

    def save(self, sector: str, new_data: pd.DataFrame, new_data_type: np.dtype):
        ss_path = os.path.join(self.feature_selection_dir, self.save_id(), sector + ".ss")
        ss = PySharedStackPlus(ss_path, dtype=new_data_type, push_enable=True)
        ss.append_from_DataFrame(new_data=new_data, new_data_type=new_data_type)
        return 0

    def main(self, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar, verbose: bool):
        sec_avlb_data = self.load_sector_available()
        model_update_days = calendar.get_last_days_in_range(bgn_date=bgn_date, end_date=end_date)
        selected_features: list[pd.DataFrame] = []
        for model_update_day in model_update_days:
            d = self.select(model_update_day, sec_avlb_data, calendar, verbose)
            selected_features.append(d)
        new_data = pd.concat(selected_features, axis=0, ignore_index=True)
        self.save(self.test.sector, new_data=new_data, new_data_type=self.save_data_type)
        return 0


"""
Part III: Wrapper for CF

"""


def process_for_feature_selection(
    test: CTestFtSlc,
    tsdb_root_dir: str,
    tsdb_user_prefix: list[str],
    freq: str,
    avlb_path: str,
    feature_selection_dir: str,
    universe: dict[str, dict[str, str]],
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    verbose: bool,
):
    selector = CFeatureSelection(
        test=test,
        tsdb_root_dir=tsdb_root_dir,
        tsdb_user_prefix=tsdb_user_prefix,
        freq=freq,
        avlb_path=avlb_path,
        feature_selection_dir=feature_selection_dir,
        universe=universe,
    )
    selector.main(bgn_date=bgn_date, end_date=end_date, calendar=calendar, verbose=verbose)
    return 0


@qtimer
def main_feature_selection(
    tests: list[CTestFtSlc],
    tsdb_root_dir: str,
    tsdb_user_prefix: list[str],
    freq: str,
    avlb_path: str,
    feature_selection_dir: str,
    universe: dict[str, dict[str, str]],
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    call_multiprocess: bool,
    processes: int,
    verbose: bool,
):
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description="[INF] Training and prediction for machine learning", total=len(tests))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for test in tests:
                    pool.apply_async(
                        process_for_feature_selection,
                        kwds={
                            "test": test,
                            "tsdb_root_dir": tsdb_root_dir,
                            "tsdb_user_prefix": tsdb_user_prefix,
                            "freq": freq,
                            "avlb_path": avlb_path,
                            "feature_selection_dir": feature_selection_dir,
                            "universe": universe,
                            "bgn_date": bgn_date,
                            "end_date": end_date,
                            "calendar": calendar,
                            "verbose": verbose,
                        },
                        callback=lambda _: pb.update(task_id=main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for test in track(tests, description="[INF] Training and prediction for machine learning"):
            process_for_feature_selection(
                test=test,
                tsdb_root_dir=tsdb_root_dir,
                tsdb_user_prefix=tsdb_user_prefix,
                freq=freq,
                avlb_path=avlb_path,
                feature_selection_dir=feature_selection_dir,
                universe=universe,
                bgn_date=bgn_date,
                end_date=end_date,
                calendar=calendar,
                verbose=verbose,
            )
    return 0


def get_feature_selection_tests(
    trn_wins: list[int], sectors: list[str], rets: list[CRet], factors_pool: list[TFactor]
) -> list[CTestFtSlc]:
    tests: list[CTestFtSlc] = []
    for trn_win, sector, ret in product(trn_wins, sectors, rets):
        test = CTestFtSlc(
            trn_win=trn_win,
            sector=sector,
            ret=ret,
            facs_pool=factors_pool,
        )
        tests.append(test)
    return tests
