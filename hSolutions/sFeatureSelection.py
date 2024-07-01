import os
import datetime as dt
import multiprocessing as mp
import numpy as np
import pandas as pd
from itertools import product
from rich.progress import track, Progress
from sklearn.feature_selection import mutual_info_regression
from hUtils.tools import qtimer, SFG, SFY, SFR, error_handler
from hUtils.instruments import parse_instrument_from_contract
from hUtils.calendar import CCalendar
from hUtils.ioPlus import PySharedStackPlus, load_tsdb
from hUtils.typeDef import CTestFtSlc, TFactor, TFactorClass, TFactorName, TReturnName, CRet, CTest


class CFeatSlcReaderAndWriter:
    def __init__(self, test: CTestFtSlc, feature_selection_dir: str):
        self.test = test
        self.feature_selection_dir = feature_selection_dir
        self.slc_fac_data: pd.DataFrame = pd.DataFrame()

    @property
    def data_settings(self) -> dict:
        return {
            "tp": np.int64,
            "trading_day": np.int32,
            "factor_class": "S40",
            "factor_name": "S40",
        }

    @property
    def save_data_type(self) -> np.dtype:
        return np.dtype([(k, v) for k, v in self.data_settings.items()])

    def save_id(self) -> str:
        return f"{self.test.ret.ret_name}-roll-W{self.test.trn_win:03d}-SlcFac"

    def load(self):
        ss_path = os.path.join(self.feature_selection_dir, self.save_id(), self.test.sector + ".ss")
        ss = PySharedStackPlus(ss_path)
        self.slc_fac_data = pd.DataFrame(ss.read_all()).set_index("trading_day")
        return 0

    def save(self, new_data: pd.DataFrame, new_data_type: np.dtype):
        ss_path = os.path.join(self.feature_selection_dir, self.save_id(), self.test.sector + ".ss")
        ss = PySharedStackPlus(ss_path, dtype=new_data_type, push_enable=True)
        ss.append_from_DataFrame(new_data=new_data, new_data_type=new_data_type)
        return 0

    def get_slc_facs(self, trading_day: np.int32) -> list[TFactor]:
        trading_day_data = self.slc_fac_data.loc[trading_day]
        res = []
        if isinstance(trading_day_data, pd.Series):
            factor_class, factor_name = trading_day_data["factor_class"], trading_day_data["factor_name"]
            fc, fn = TFactorClass(factor_class.decode("utf-8")), TFactorName(factor_name.decode("utf-8"))
            res.append(TFactor((fc, fn)))
        elif isinstance(trading_day_data, pd.DataFrame):
            for factor_class, factor_name in zip(trading_day_data["factor_class"], trading_day_data["factor_name"]):
                fc, fn = TFactorClass(factor_class.decode("utf-8")), TFactorName(factor_name.decode("utf-8"))
                res.append(TFactor((fc, fn)))
        else:
            raise TypeError(f"type of trading_day_data is {type(trading_day_data)}")
        return res


class CFeatSlc(CFeatSlcReaderAndWriter):
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
        facs_pool: list[TFactor],
    ):
        super().__init__(test, feature_selection_dir)
        self.tsdb_root_dir = tsdb_root_dir
        self.tsdb_user_prefix = tsdb_user_prefix
        self.freq = freq

        self.avlb_path = avlb_path
        self.universe = universe
        self.facs_pool = facs_pool
        self.__x_cols: list[TFactorName] = [n for _, n in self.facs_pool]
        self.mapper_name_to_class: dict[TFactorName, TFactorClass] = {n: c for c, n in self.facs_pool}

    @property
    def x_cols(self) -> list[TFactorName]:
        return self.__x_cols

    @property
    def y_col(self) -> TReturnName:
        return self.test.ret.ret_name

    def load_x(self, bgn_date: np.int32, end_date: np.int32) -> pd.DataFrame:
        usr_prefix = ".".join(self.tsdb_user_prefix)
        value_columns, rename_mapper = [], {}
        for factor_class, factor_name in self.facs_pool:
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

    def core(self, x_data: pd.DataFrame, y_data: pd.Series, trading_day: np.int32) -> list[TFactorName]:
        raise NotImplementedError

    def get_factor_class(self, factor_names: list[TFactorName]) -> list[TFactorClass]:
        return [self.mapper_name_to_class[n] for n in factor_names]

    def get_selected_feats(
        self, trading_day: np.int32, factor_class: list[TFactorClass], factor_names: list[TFactorName]
    ) -> pd.DataFrame:
        if factor_names:
            tp = dt.datetime.strptime(f"{trading_day} 16:00:00", "%Y%m%d %H:%M:%S").timestamp() * 10**9
            selected_feats = pd.DataFrame(
                {
                    "tp": tp,
                    "trading_day": trading_day,
                    "factor_class": factor_class,
                    "factor_names": factor_names,
                }
            )
        else:
            print(
                f"[{SFR('WRN')}] No features are selected @ {SFG(trading_day)} for {self.test.sector} {self.test.trn_win} {self.test.ret.ret_name}"
            )
            selected_feats = pd.DataFrame(columns=["tp", "trading_day", "factor_class", "factor_names"])
        return selected_feats

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
        factor_names = self.core(x_data=X, y_data=y, trading_day=trn_e_date)
        factor_class = self.get_factor_class(factor_names=factor_names)
        selected_feats = self.get_selected_feats(trn_e_date, factor_class=factor_class, factor_names=factor_names)
        if verbose:
            print(
                f"[INF] Feature selection @ {SFG(int(model_update_day))}, "
                f"factor selected @ {SFG(int(trn_e_date))}, "
                f"using train data @ [{SFG(int(trn_b_date))},{SFG(int(trn_e_date))}], "
                f"save as {SFG(int(model_update_month))}"
            )
        return selected_feats

    def main(self, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar, verbose: bool):
        sec_avlb_data = self.load_sector_available()
        model_update_days = calendar.get_last_days_in_range(bgn_date=bgn_date, end_date=end_date)
        selected_features: list[pd.DataFrame] = []
        for model_update_day in model_update_days:
            slc_feats = self.select(model_update_day, sec_avlb_data, calendar, verbose)
            if not slc_feats.empty:
                selected_features.append(slc_feats)
        new_data = pd.concat(selected_features, axis=0, ignore_index=True)
        self.save(new_data=new_data, new_data_type=self.save_data_type)
        return 0


class CFeatSlcMutInf(CFeatSlc):
    def __init__(
        self,
        threshold: float,
        min_feats: int,
        test: CTestFtSlc,
        tsdb_root_dir: str,
        tsdb_user_prefix: list[str],
        freq: str,
        avlb_path: str,
        feature_selection_dir: str,
        universe: dict[str, dict[str, str]],
        facs_pool: list[TFactor],
    ):
        self.threshold = threshold
        self.min_feats = min_feats
        super().__init__(
            test=test,
            tsdb_root_dir=tsdb_root_dir,
            tsdb_user_prefix=tsdb_user_prefix,
            freq=freq,
            avlb_path=avlb_path,
            feature_selection_dir=feature_selection_dir,
            universe=universe,
            facs_pool=facs_pool,
        )

    def core(self, x_data: pd.DataFrame, y_data: pd.Series, trading_day: np.int32) -> list[TFactorName]:
        __minimum_score = 1e-4
        importance = mutual_info_regression(X=x_data, y=y_data, random_state=self.RANDOM_STATE)
        feat_importance = pd.Series(data=importance, index=x_data.columns).sort_values(ascending=False)
        # if False:
        #     corr = [np.corrcoef(x_data[col], y_data)[0, 1] for col in x_data.columns]
        #     feat_corr = pd.Series(data=corr, index=x_data.columns).sort_values(ascending=False)
        #     df = pd.DataFrame({"mutual_info": feat_importance, "corr": feat_corr}).sort_values(
        #         by=["mutual_info", "corr"], ascending=False
        #     )

        if len(available_feats := feat_importance[feat_importance >= __minimum_score]) < self.min_feats:
            return [TFactorName(z) for z in available_feats.index]

        t, i = self.threshold, 0
        while len(selected_feats := feat_importance[feat_importance >= t]) < self.min_feats:
            t, i = t * 0.8, i + 1
        if i > 0:
            print(
                f"[INF] After {SFY(i)} times iteration {SFY(f'{len(selected_feats):>2d}')} features are selected, "
                f"{SFY(self.test.sector)}-{SFY(self.test.trn_win)}-{SFY(trading_day)}-{SFY(self.test.ret.desc)}"
            )
        return [TFactorName(z) for z in selected_feats.index]


def process_for_feature_selection(
    threshold: float,
    min_feats: int,
    test: CTestFtSlc,
    facs_pool: list[TFactor],
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
    selector = CFeatSlcMutInf(
        threshold=threshold,
        min_feats=min_feats,
        test=test,
        tsdb_root_dir=tsdb_root_dir,
        tsdb_user_prefix=tsdb_user_prefix,
        freq=freq,
        avlb_path=avlb_path,
        feature_selection_dir=feature_selection_dir,
        universe=universe,
        facs_pool=facs_pool,
    )
    selector.main(bgn_date=bgn_date, end_date=end_date, calendar=calendar, verbose=verbose)
    return 0


@qtimer
def main_feature_selection(
    threshold: float,
    min_feats: int,
    tests: list[CTestFtSlc],
    facs_pool: list[TFactor],
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
            main_task = pb.add_task(description="[INF] Selecting features ...", total=len(tests))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                for test in tests:
                    pool.apply_async(
                        process_for_feature_selection,
                        kwds={
                            "threshold": threshold,
                            "min_feats": min_feats,
                            "test": test,
                            "facs_pool": facs_pool,
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
        for test in track(tests, description="[INF] Selecting features ..."):
            # for test in tests:
            process_for_feature_selection(
                threshold=threshold,
                min_feats=min_feats,
                test=test,
                facs_pool=facs_pool,
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


def get_feature_selection_tests(trn_wins: list[int], sectors: list[str], rets: list[CRet]) -> list[CTestFtSlc]:
    tests: list[CTestFtSlc] = []
    for trn_win, sector, ret in product(trn_wins, sectors, rets):
        test = CTestFtSlc(
            trn_win=trn_win,
            sector=sector,
            ret=ret,
        )
        tests.append(test)
    return tests
