import datetime as dt
import numpy as np
import pandas as pd
from pyqts.PySharedStack import PySharedStack
from pyqts.pytsdb import PyTsdb
from hUtils.calendar import CCalendar
from hUtils.tools import SFG, SFY


class PySharedStackPlus(PySharedStack):
    def __init__(self, pth: str, dtype: np.dtype = None, push_enable=False):  # type: ignore
        self.pth = pth
        super().__init__(pth=pth, dtype=dtype, push_enable=push_enable)

    @property
    def empty(self) -> bool:
        return self.m_ss.size() == 0

    def tail(self, n: int = 1):
        sz = self.m_ss.size()
        res = np.empty(dtype=self.dtype, shape=[n])
        self.m_ss.read(sz - n, sz, res)
        return res

    @property
    def last_date(self) -> np.int32:
        return self.tail()[0]["trading_day"]

    @property
    def last_idx(self) -> int:
        return self.tail()[0]["idx_tick_end"]

    @property
    def last_oi(self) -> int:
        return self.tail()[0]["oi_close"]

    @property
    def last_barno(self) -> int:
        return self.tail()[0]["barno"]

    @property
    def last_nav(self) -> np.float64:
        return self.tail()[0]["nav"]

    def last_price(self, price: str = "closeM") -> np.float64:
        return self.tail()[0][price]

    def last_price_not_nan(self, price: str = "closeM") -> np.float64:
        df = pd.DataFrame(self.read_all())
        s = df[price].dropna()
        return np.float64(np.nan) if s.empty else s.iloc[-1]

    def check_daily_continuous(self, coming_next_date: np.int32, calendar: CCalendar) -> int:
        if self.empty:
            return 0
        expected_next_date = calendar.get_next_date(self.last_date, 1)
        if expected_next_date == coming_next_date:
            return 0
        elif expected_next_date > coming_next_date:
            print(
                f"[INF] Data is not continuous in {self.pth}\n"
                f"[INF] "
                f"LAST DATE in this share stack is {SFG(str(self.last_date))}, "
                f"and EXPECTED NEXT DATE is {SFG(str(expected_next_date))}, "
                f"but FIRST DATE in new data is {SFG(str(coming_next_date))}, "
                f"some days maybe {SFY('duplicated')}."
            )
            return 1
        else:  # expected_next_date < coming_next_date:
            print(
                f"[INF] Data is not continuous in {self.pth}\n"
                f"[INF] "
                f"LAST DATE in this share stack is {SFG(str(self.last_date))}, "
                f"and EXPECTED NEXT DATE is {SFG(str(expected_next_date))}, "
                f"but FIRST DATE in new data is {SFG(str(coming_next_date))}, "
                f"some days maybe {SFY('missing')}."
            )
            return -1

    def append_from_DataFrame(self, new_data: pd.DataFrame, new_data_type: np.dtype):
        new_data_ary = np.array(list(new_data.itertuples(index=False)), dtype=new_data_type)
        self.push_back(data=new_data_ary)
        return 0


class CTsdbPlus(PyTsdb):
    DTYPES = {"status": np.int32, "bar_no": np.int32, "uid": "S8", "types": "S8", "hot": np.int32}

    def read_by_date(self, value_columns: list[str], trade_date: str, freq: str = "d01e") -> pd.DataFrame:
        tp_beg = trade_date + " 08:00:00.000"
        tp_end = trade_date + " 17:00:00.000"
        tsdb_data = self.read_columns(freq, value_columns, tp_beg, tp_end)
        df = pd.DataFrame(tsdb_data)
        return df

    def read_range(self, value_columns: list[str], beg_date: str, end_date: str, freq: str = "d01e") -> pd.DataFrame:
        if freq in ["d01e"]:
            tp_beg = beg_date + " 08:00:00.000"
            tp_end = end_date + " 17:00:00.000"
            tsdb_data = self.read_columns(freq, value_columns, tp_beg, tp_end, dtypes=self.DTYPES)
            df = pd.DataFrame(tsdb_data)
        elif freq in ["d01b", "m01e", "m05e", "m15e"]:
            dt_bgn = dt.datetime.strptime(beg_date, "%Y%m%d") - dt.timedelta(days=20)
            tp_beg = dt_bgn.strftime("%Y%m%d %H:%M:%S.%f")[:-3]
            tp_end = end_date + " 17:00:00.000"
            freq_dtypes = {**self.DTYPES, **{"amount": np.float64, "vol": np.int32}}
            tsdb_data = self.read_columns(freq, value_columns, tp_beg, tp_end, dtypes=freq_dtypes)
            df = pd.DataFrame(tsdb_data)
            filter_days = (df["trading_day"] >= int(beg_date)) & (df["trading_day"] <= int(end_date))
            df = df[filter_days].reset_index(drop=True)
        else:
            raise ValueError(f"[ERR] Read range is NOT IMPLEMENTED for this freq = {SFG(freq)}")
        return df


def load_tsdb(
    tsdb_root_dir: str, freq: str, value_columns: list[str], bgn_date: np.int32, end_date: np.int32
) -> pd.DataFrame:
    db_reader = CTsdbPlus(tsdb_root_dir)
    tsdb_data = db_reader.read_range(
        value_columns=value_columns,
        beg_date=str(bgn_date),
        end_date=str(end_date),
        freq=freq.replace("/", ""),
    )
    return tsdb_data
