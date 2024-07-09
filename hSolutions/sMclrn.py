import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import skops.io as sio
import lightgbm as lgb
import xgboost as xgb
from rich.progress import track, Progress
from sklearn.linear_model import Ridge
from hUtils.tools import qtimer, SFG, SFY, SFR, check_and_mkdir, error_handler
from hUtils.instruments import parse_instrument_from_contract
from hUtils.calendar import CCalendar
from hUtils.ioPlus import PySharedStackPlus, load_tsdb
from hUtils.typeDef import CTest, CTestFtSlc, TFactor
from hSolutions.sFeatureSelection import CFeatSlcReaderAndWriter


"""
Part II: Class for Machine Learning
"""


class CMclrn:
    XY_INDEX = ["tp", "trading_day", "ticker", "instru"]
    RANDOM_STATE = 0

    def __init__(
        self,
        using_instru: bool,
        test: CTest,
        tsdb_root_dir: str,
        tsdb_user_prefix: list[str],
        freq: str,
        avlb_path: str,
        mclrn_dir: str,
        prediction_dir: str,
        universe: dict[str, dict[str, str]],
    ):
        self.using_instru = using_instru
        self.__x_cols: list[str] = []
        self.prototype = NotImplemented
        self.fitted_estimator = NotImplemented

        self.test = test

        self.tsdb_root_dir = tsdb_root_dir
        self.tsdb_user_prefix = tsdb_user_prefix
        self.freq = freq

        self.avlb_path = avlb_path
        self.mclrn_dir = mclrn_dir
        self.prediction_dir = prediction_dir
        self.universe = universe

    @property
    def x_cols(self) -> list[str]:
        return self.__x_cols

    @x_cols.setter
    def x_cols(self, x_cols: list[str]):
        self.__x_cols = x_cols

    @property
    def y_col(self) -> str:
        return self.test.ret.ret_name

    @property
    def data_settings_prd(self) -> dict:
        return {
            "tp": np.int64,
            "trading_day": np.int32,
            "ticker": "S8",
            self.test.ret.ret_name: np.float32,
        }

    @property
    def data_types_prd(self) -> np.dtype:
        return np.dtype([(k, v) for k, v in self.data_settings_prd.items()])

    def reset_estimator(self):
        self.fitted_estimator = None
        return 0

    def get_slc_facs(self, trading_day: np.int32) -> list[TFactor]:
        raise NotImplementedError

    def set_x_cols(self, slc_facs: list[TFactor]):
        self.__x_cols = [n for _, n in slc_facs]

    def load_x(self, bgn_date: np.int32, end_date: np.int32, slc_facs: list[TFactor]) -> pd.DataFrame:
        usr_prefix = ".".join(self.tsdb_user_prefix)
        value_columns, rename_mapper = [], {}
        for factor_class, factor_name in slc_facs:
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

    def fit_estimator(self, x_data: pd.DataFrame, y_data: pd.Series):
        if self.using_instru:
            X, y = x_data.reset_index(level="instru"), y_data
            X["instru"] = X["instru"].astype("category")
        else:
            X, y = x_data.values, y_data.values
        self.fitted_estimator = self.prototype.fit(X, y)
        return 0

    def save_model(self, month_id: np.int32):
        model_file = f"{self.test.save_tag_mdl}.skops"
        check_and_mkdir(month_dir := os.path.join(self.mclrn_dir, str(month_id)))
        model_path = os.path.join(month_dir, model_file)
        sio.dump(self.fitted_estimator, model_path)
        return 0

    def load_model(self, month_id: np.int32, verbose: bool) -> bool:
        model_file = f"{self.test.save_tag_mdl}.skops"
        model_path = os.path.join(self.mclrn_dir, str(month_id), model_file)
        if os.path.exists(model_path):
            self.fitted_estimator = sio.load(model_path, trusted=True)
            return True
        else:
            if verbose:
                print(f"[INF] No model file for {SFY(self.test.save_tag_mdl)} at {SFY(int(month_id))}")
            return False

    def apply_estimator(self, x_data: pd.DataFrame) -> pd.Series:
        if self.using_instru:
            X = x_data.reset_index(level="instru")
            X["instru"] = X["instru"].astype("category")
        else:
            X = x_data.values
        pred = self.fitted_estimator.predict(X=X)  # type:ignore
        return pd.Series(data=pred, name=self.y_col, index=x_data.index)

    def train(self, model_update_day: np.int32, sec_avlb_data: pd.DataFrame, calendar: CCalendar, verbose: bool):
        model_update_month = model_update_day // 100
        trn_b_date = calendar.get_next_date(model_update_day, shift=-self.test.ret.shift - self.test.trn_win + 1)
        trn_e_date = calendar.get_next_date(model_update_day, shift=-self.test.ret.shift)
        slc_facs = self.get_slc_facs(trading_day=trn_e_date)
        self.set_x_cols(slc_facs=slc_facs)

        sec_avlb_data = self.truncate_avlb(sec_avlb_data, trn_b_date, trn_e_date)
        x_data, y_data = self.load_x(trn_b_date, trn_e_date, slc_facs), self.load_y(trn_b_date, trn_e_date)
        x_data, y_data = self.filter_by_sector(x_data, sec_avlb_data), self.filter_by_sector(y_data, sec_avlb_data)
        x_data, y_data = self.set_index(x_data), self.set_index(y_data)
        aligned_data = self.aligned_xy(x_data, y_data)
        aligned_data = self.drop_and_fill_nan(aligned_data)
        X, y = self.get_X_y(aligned_data=aligned_data)
        self.fit_estimator(x_data=X, y_data=y)
        self.save_model(month_id=model_update_month)
        if verbose:
            print(
                f"[INF] Train model @ {SFG(int(model_update_day))}, "
                f"factor selected @ {SFG(int(trn_e_date))}, "
                f"using train data @ [{SFG(int(trn_b_date))},{SFG(int(trn_e_date))}], "
                f"save as {SFG(int(model_update_month))}"
            )
        return 0

    def process_trn(self, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar, verbose: bool):
        sec_avlb_data = self.load_sector_available()
        model_update_days = calendar.get_last_days_in_range(bgn_date=bgn_date, end_date=end_date)
        for model_update_day in model_update_days:
            self.train(model_update_day, sec_avlb_data, calendar, verbose)
        return 0

    def predict(
        self,
        prd_month_id: np.int32,
        prd_month_days: list[np.int32],
        avlb_data: pd.DataFrame,
        calendar: CCalendar,
        verbose: bool,
    ) -> pd.Series:
        trn_month_id = calendar.get_next_month(prd_month_id, -1)
        self.reset_estimator()
        if self.load_model(month_id=trn_month_id, verbose=verbose):
            model_update_day = calendar.get_last_day_of_month(trn_month_id)
            trn_e_date = calendar.get_next_date(model_update_day, shift=-self.test.ret.shift)
            slc_facs = self.get_slc_facs(trading_day=trn_e_date)
            self.set_x_cols(slc_facs=slc_facs)

            prd_b_date, prd_e_date = prd_month_days[0], prd_month_days[-1]
            sec_avlb_data = self.truncate_avlb(avlb_data, prd_b_date, prd_e_date)
            x_data = self.load_x(prd_b_date, prd_e_date, slc_facs)
            x_data = self.filter_by_sector(x_data, sec_avlb_data)
            x_data = self.set_index(x_data)
            x_data = self.drop_and_fill_nan(x_data)
            y_h_data = self.apply_estimator(x_data=x_data)
            if verbose:
                print(
                    f"[INF] Call model @ {SFG(int(model_update_day))}, "
                    f"factor selected @ {SFG(int(trn_e_date))}, "
                    f"prediction @ [{SFG(int(prd_b_date))},{SFG(int(prd_e_date))}], "
                    f"load model from {SFG(int(trn_month_id))}"
                )
            return y_h_data.astype(np.float64)
        else:
            return pd.Series(dtype=np.float64)

    def process_prd(self, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar, verbose: bool) -> pd.DataFrame:
        avlb_data = self.load_sector_available()
        months_groups = calendar.split_by_month(dates=calendar.get_iter_list(bgn_date, end_date))
        pred_res: list[pd.Series] = []
        for prd_month_id, prd_month_days in months_groups.items():
            month_prediction = self.predict(prd_month_id, prd_month_days, avlb_data, calendar, verbose)
            pred_res.append(month_prediction)
        prediction = pd.concat(pred_res, axis=0, ignore_index=False)
        prediction.index = pd.MultiIndex.from_tuples(prediction.index, names=self.XY_INDEX)
        sorted_prediction = prediction.reset_index().sort_values(["tp", "trading_day", "ticker"])
        return sorted_prediction

    def process_save_prediction(self, prediction: pd.DataFrame, calendar: CCalendar):
        for instru, instru_df in prediction.groupby(by="instru"):
            ss_path = os.path.join(self.prediction_dir, self.test.save_tag_prd, f"{instru}.ss")
            ss = PySharedStackPlus(ss_path, dtype=self.data_types_prd, push_enable=True)
            if ss.check_daily_continuous(instru_df["trading_day"].iloc[0], calendar=calendar) == 0:
                new_data = instru_df.drop(axis=1, labels=["instru"])
                ss.append_from_DataFrame(new_data=new_data, new_data_type=self.data_types_prd)
        return 0

    def main_mclrn_model(self, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar, verbose: bool):
        self.process_trn(bgn_date, end_date, calendar, verbose)
        prediction = self.process_prd(bgn_date, end_date, calendar, verbose)
        self.process_save_prediction(prediction, calendar)
        return 0


class CMclrnFromFeatureSelection(CMclrn):
    def __init__(
        self,
        using_instru: bool,
        test: CTest,
        tsdb_root_dir: str,
        tsdb_user_prefix: list[str],
        freq: str,
        avlb_path: str,
        mclrn_dir: str,
        prediction_dir: str,
        feature_selection_dir: str,
        universe: dict[str, dict[str, str]],
    ):
        super().__init__(
            using_instru=using_instru,
            test=test,
            tsdb_root_dir=tsdb_root_dir,
            tsdb_user_prefix=tsdb_user_prefix,
            freq=freq,
            avlb_path=avlb_path,
            mclrn_dir=mclrn_dir,
            prediction_dir=prediction_dir,
            universe=universe,
        )
        test_slcFac = CTestFtSlc(trn_win=test.trn_win, sector=test.sector, ret=test.ret)
        self.slc_fac_reader = CFeatSlcReaderAndWriter(test=test_slcFac, feature_selection_dir=feature_selection_dir)
        self.slc_fac_reader.load()

    def get_slc_facs(self, trading_day: np.int32) -> list[TFactor]:
        return self.slc_fac_reader.get_slc_facs(trading_day=trading_day)


class CMclrnRidge(CMclrnFromFeatureSelection):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(using_instru=False, **kwargs)
        self.prototype = Ridge(alpha=alpha, fit_intercept=False)


class CMclrnLGBM(CMclrnFromFeatureSelection):
    def __init__(
        self,
        boosting_type: str,
        metric: str,
        max_depth: int,
        num_leaves: int,
        learning_rate: float,
        n_estimators: int,
        min_child_samples: int,
        max_bin: int,
        **kwargs,
    ):
        super().__init__(using_instru=True, **kwargs)
        self.prototype = lgb.LGBMRegressor(
            boosting_type=boosting_type,
            metric=metric,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            max_bin=max_bin,
            # other fixed parameters
            force_row_wise=True,
            verbose=-1,
            random_state=self.RANDOM_STATE,
        )


class CMclrnXGB(CMclrnFromFeatureSelection):
    def __init__(
        self,
        booster: str,
        n_estimators: int,
        max_depth: int,
        max_leaves: int,
        grow_policy: str,
        learning_rate: float,
        objective: str,
        **kwargs,
    ):
        super().__init__(using_instru=False, **kwargs)
        self.prototype = xgb.XGBRegressor(
            booster=booster,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_leaves=max_leaves,
            grow_policy=grow_policy,
            learning_rate=learning_rate,
            objective=objective,
            # other fixed parameters
            verbosity=0,
            random_state=self.RANDOM_STATE,
            # nthread=6,
        )


"""
Part III: Wrapper for CMclrn

"""


def process_for_cMclrn(
    test: CTest,
    tsdb_root_dir: str,
    tsdb_user_prefix: list[str],
    freq: str,
    avlb_path: str,
    mclrn_dir: str,
    prediction_dir: str,
    feature_selection_dir: str,
    universe: dict[str, dict[str, str]],
    bgn_date: np.int32,
    end_date: np.int32,
    calendar: CCalendar,
    verbose: bool,
):
    x: dict[str, type[CMclrnRidge] | type[CMclrnLGBM] | type[CMclrnXGB]] = {
        "Ridge": CMclrnRidge,
        "LGBM": CMclrnLGBM,
        "XGB": CMclrnXGB,
    }
    if not (mclrn_type := x.get(test.model.model_type)):
        raise ValueError(f"model type = {test.model.model_type} is wrong")

    mclrn = mclrn_type(
        test=test,
        tsdb_root_dir=tsdb_root_dir,
        tsdb_user_prefix=tsdb_user_prefix,
        freq=freq,
        avlb_path=avlb_path,
        mclrn_dir=mclrn_dir,
        prediction_dir=prediction_dir,
        feature_selection_dir=feature_selection_dir,
        universe=universe,
        **test.model.model_args,
    )
    if isinstance(mclrn, CMclrnXGB):
        os.environ['OMP_NUM_THREADS'] = "8"

    mclrn.main_mclrn_model(bgn_date=bgn_date, end_date=end_date, calendar=calendar, verbose=verbose)
    return 0


@qtimer
def main_train_and_predict(
    tests: list[CTest],
    tsdb_root_dir: str,
    tsdb_user_prefix: list[str],
    freq: str,
    avlb_path: str,
    mclrn_dir: str,
    prediction_dir: str,
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
                        process_for_cMclrn,
                        kwds={
                            "test": test,
                            "tsdb_root_dir": tsdb_root_dir,
                            "tsdb_user_prefix": tsdb_user_prefix,
                            "freq": freq,
                            "avlb_path": avlb_path,
                            "mclrn_dir": mclrn_dir,
                            "prediction_dir": prediction_dir,
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
            # for test in tests:
            process_for_cMclrn(
                test=test,
                tsdb_root_dir=tsdb_root_dir,
                tsdb_user_prefix=tsdb_user_prefix,
                freq=freq,
                avlb_path=avlb_path,
                mclrn_dir=mclrn_dir,
                prediction_dir=prediction_dir,
                feature_selection_dir=feature_selection_dir,
                universe=universe,
                bgn_date=bgn_date,
                end_date=end_date,
                calendar=calendar,
                verbose=verbose,
            )
    return 0
