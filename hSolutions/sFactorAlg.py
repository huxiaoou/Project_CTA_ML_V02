import dataclasses
import numpy as np
import pandas as pd
import itertools as ittl
from hUtils.calendar import CCalendar
from hUtils.typeDef import TFactorClass, TFactorNames, TFactorName, TFactorClassAndNames
from hUtils.typeDef import TFactorComb
from hSolutions.sFactor import CFactor

"""
-----------------------
Part I: Some math tools
-----------------------
"""


def cal_rolling_corr(df: pd.DataFrame, x: str, y: str, rolling_window: int) -> pd.Series:
    df["xy"] = (df[x] * df[y]).rolling(window=rolling_window).mean()
    df["xx"] = (df[x] * df[x]).rolling(window=rolling_window).mean()
    df["yy"] = (df[y] * df[y]).rolling(window=rolling_window).mean()
    df["x"] = df[x].rolling(window=rolling_window).mean()
    df["y"] = df[y].rolling(window=rolling_window).mean()

    df["cov_xy"] = df["xy"] - df["x"] * df["y"]
    df["cov_xx"] = df["xx"] - df["x"] * df["x"]
    df["cov_yy"] = df["yy"] - df["y"] * df["y"]

    # due to float number precision, cov_xx or cov_yy could be slightly negative
    df.loc[np.abs(df["cov_xx"]) <= 1e-10, "cov_xx"] = 0
    df.loc[np.abs(df["cov_yy"]) <= 1e-10, "cov_yy"] = 0

    df["sqrt_cov_xx_yy"] = np.sqrt(df["cov_xx"] * df["cov_yy"])
    s = df[["cov_xy", "sqrt_cov_xx_yy"]].apply(
        lambda z: 0 if z["sqrt_cov_xx_yy"] == 0 else z["cov_xy"] / z["sqrt_cov_xx_yy"], axis=1
    )
    return s


def cal_rolling_beta(df: pd.DataFrame, x: str, y: str, rolling_window: int) -> pd.Series:
    df["xy"] = (df[x] * df[y]).rolling(window=rolling_window).mean()
    df["xx"] = (df[x] * df[x]).rolling(window=rolling_window).mean()
    df["x"] = df[x].rolling(window=rolling_window).mean()
    df["y"] = df[y].rolling(window=rolling_window).mean()
    df["cov_xy"] = df["xy"] - df["x"] * df["y"]
    df["cov_xx"] = df["xx"] - df["x"] * df["x"]
    s = df["cov_xy"] / df["cov_xx"]
    return s


def cal_top_corr(sub_data: pd.DataFrame, x: str, y: str, sort_var: str, top_size: int, ascending: bool = False):
    sorted_data = sub_data.sort_values(by=sort_var, ascending=ascending)
    top_data = sorted_data.head(top_size)
    r = top_data[[x, y]].corr(method="spearman").at[x, y]
    return r


def auto_weight_sum(x: pd.Series) -> float:
    weight = x.abs() / x.abs().sum()
    return x @ weight


"""
---------------------------------------
Part II: Class for factor configuration
---------------------------------------
"""


@dataclasses.dataclass(frozen=True)
class CCfgFactor:
    @property
    def factor_class(self) -> TFactorClass:
        raise NotImplementedError

    @property
    def factor_names(self) -> TFactorNames:
        raise NotImplementedError

    def get_raw_class_and_names(self) -> TFactorClassAndNames:
        return TFactorClassAndNames((self.factor_class, self.factor_names))

    def get_neu_class_and_names(self) -> TFactorClassAndNames:
        neu_class = TFactorClass(f"{self.factor_class}-NEU")
        neu_names = TFactorNames([TFactorName(f"{_}-NEU") for _ in self.factor_names])
        return TFactorClassAndNames((neu_class, neu_names))

    def get_combs_raw(self) -> list[TFactorComb]:
        factor_class, factor_names = self.get_raw_class_and_names()
        return [TFactorComb((factor_class, factor_names, "factors_by_instru"))]

    def get_combs_neu(self) -> list[TFactorComb]:
        factor_class, factor_names = self.get_neu_class_and_names()
        return [TFactorComb((factor_class, factor_names, "neutral_by_instru"))]

    def get_combs(self) -> list[TFactorComb]:
        return self.get_combs_raw() + self.get_combs_neu()


# cfg for factors
@dataclasses.dataclass(frozen=True)
class CCfgFactorMTM(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("MTM")

    @property
    def factor_names(self) -> TFactorNames:
        return TFactorNames([TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins])


@dataclasses.dataclass(frozen=True)
class CCfgFactorSKEW(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("SKEW")

    @property
    def factor_names(self) -> TFactorNames:
        return TFactorNames([TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins])


@dataclasses.dataclass(frozen=True)
class CCfgFactorRS(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("RS")

    @property
    def factor_names(self) -> TFactorNames:
        rspa = [TFactorName(f"{self.factor_class}PA{w:03d}") for w in self.wins]
        rsla = [TFactorName(f"{self.factor_class}LA{w:03d}") for w in self.wins]
        return TFactorNames(rspa + rsla)


@dataclasses.dataclass(frozen=True)
class CCfgFactorBASIS(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("BASIS")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        n1 = [TFactorName(f"{self.factor_class}D{w:03d}") for w in self.wins]
        return TFactorNames(n0 + n1)


@dataclasses.dataclass(frozen=True)
class CCfgFactorTS(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("TS")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        n1 = [TFactorName(f"{self.factor_class}D{w:03d}") for w in self.wins]
        return TFactorNames(n0 + n1)


@dataclasses.dataclass(frozen=True)
class CCfgFactorS0BETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("S0BETA")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        n1 = [TFactorName(f"{self.factor_class}{self.wins[0]:03d}D{w:03d}") for w in self.wins[1:]]
        # n2 = [TFactorName(f"{self.factor_class}{w:03d}RES") for w in self.wins]
        # n3 = [TFactorName(f"{self.factor_class}{w:03d}RESSTD") for w in self.wins]
        return TFactorNames(n0 + n1)


@dataclasses.dataclass(frozen=True)
class CCfgFactorS1BETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("S1BETA")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        n1 = [TFactorName(f"{self.factor_class}{self.wins[0]:03d}D{w:03d}") for w in self.wins[1:]]
        # n2 = [TFactorName(f"{self.factor_class}{w:03d}RES") for w in self.wins]
        # n3 = [TFactorName(f"{self.factor_class}{w:03d}RESSTD") for w in self.wins]
        return TFactorNames(n0 + n1)


@dataclasses.dataclass(frozen=True)
class CCfgFactorCBETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CBETA")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        n1 = [TFactorName(f"{self.factor_class}{self.wins[0]:03d}D{w:03d}") for w in self.wins[1:]]
        # n2 = [TFactorName(f"{self.factor_class}{w:03d}RES") for w in self.wins]
        # n3 = [TFactorName(f"{self.factor_class}{w:03d}RESSTD") for w in self.wins]
        return TFactorNames(n0 + n1)


@dataclasses.dataclass(frozen=True)
class CCfgFactorIBETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("IBETA")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        n1 = [TFactorName(f"{self.factor_class}{self.wins[0]:03d}D{w:03d}") for w in self.wins[1:]]
        # n2 = [TFactorName(f"{self.factor_class}{w:03d}RES") for w in self.wins]
        # n3 = [TFactorName(f"{self.factor_class}{w:03d}RESSTD") for w in self.wins]
        return TFactorNames(n0 + n1)


@dataclasses.dataclass(frozen=True)
class CCfgFactorPBETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("PBETA")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        n1 = [TFactorName(f"{self.factor_class}{self.wins[0]:03d}D{w:03d}") for w in self.wins[1:]]
        # n2 = [TFactorName(f"{self.factor_class}{w:03d}RES") for w in self.wins]
        # n3 = [TFactorName(f"{self.factor_class}{w:03d}RESSTD") for w in self.wins]
        return TFactorNames(n0 + n1)


@dataclasses.dataclass(frozen=True)
class CCfgFactorCTP(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CTP")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t*10):02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclasses.dataclass(frozen=True)
class CCfgFactorCTR(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CTR")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t*10):02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclasses.dataclass(frozen=True)
class CCfgFactorCVP(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CVP")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t*10):02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclasses.dataclass(frozen=True)
class CCfgFactorCVR(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CVR")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t*10):02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclasses.dataclass(frozen=True)
class CCfgFactorCSP(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CSP")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t*10):02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclasses.dataclass(frozen=True)
class CCfgFactorCSR(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CSR")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t*10):02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclasses.dataclass(frozen=True)
class CCfgFactorNOI(CCfgFactor):
    wins: list[int]
    tops: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("NOI")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{t:02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclasses.dataclass(frozen=True)
class CCfgFactorNDOI(CCfgFactor):
    wins: list[int]
    tops: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("NDOI")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{t:02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclasses.dataclass(frozen=True)
class CCfgFactorWNOI(CCfgFactor):
    wins: list[int]
    tops: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("WNOI")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{t:02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclasses.dataclass(frozen=True)
class CCfgFactorWNDOI(CCfgFactor):
    wins: list[int]
    tops: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("WNDOI")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{t:02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclasses.dataclass(frozen=True)
class CCfgFactorAMP(CCfgFactor):
    wins: list[int]
    lbds: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("AMP")

    @property
    def factor_names(self) -> TFactorNames:
        nh = [
            TFactorName(f"{self.factor_class}{w:03d}T{int(l*10):02d}H") for w, l in ittl.product(self.wins, self.lbds)
        ]
        nl = [
            TFactorName(f"{self.factor_class}{w:03d}T{int(l*10):02d}L") for w, l in ittl.product(self.wins, self.lbds)
        ]
        nd = [
            TFactorName(f"{self.factor_class}{w:03d}T{int(l*10):02d}D") for w, l in ittl.product(self.wins, self.lbds)
        ]
        return TFactorNames(nh + nl + nd)


@dataclasses.dataclass(frozen=True)
class CCfgFactorEXR(CCfgFactor):
    freq: str
    wins: list[int]
    dfts: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("EXR")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        n1 = [TFactorName(f"DXR{w:03d}D{d:02d}") for w, d in ittl.product(self.wins, self.dfts)]
        n2 = [TFactorName(f"AXR{w:03d}D{d:02d}") for w, d in ittl.product(self.wins, self.dfts)]
        return TFactorNames(n0 + n1 + n2)


@dataclasses.dataclass(frozen=True)
class CCfgFactorSMT(CCfgFactor):
    freq: str
    wins: list[int]
    lbds: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("SMT")

    @property
    def factor_names(self) -> TFactorNames:
        n_prc = [
            TFactorName(f"{self.factor_class}{w:03d}T{int(l*10):02d}P") for w, l in ittl.product(self.wins, self.lbds)
        ]
        n_ret = [
            TFactorName(f"{self.factor_class}{w:03d}T{int(l*10):02d}R") for w, l in ittl.product(self.wins, self.lbds)
        ]
        return TFactorNames(n_prc + n_ret)


@dataclasses.dataclass(frozen=True)
class CCfgFactorRWTC(CCfgFactor):
    freq: str
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("RWTC")

    @property
    def factor_names(self) -> TFactorNames:
        nu = [TFactorName(f"{self.factor_class}{w:03d}U") for w in self.wins]
        nd = [TFactorName(f"{self.factor_class}{w:03d}D") for w in self.wins]
        nt = [TFactorName(f"{self.factor_class}{w:03d}T") for w in self.wins]
        nv = [TFactorName(f"{self.factor_class}{w:03d}V") for w in self.wins]
        return TFactorNames(nu + nd + nt + nv)


# ---------------------------------
@dataclasses.dataclass(frozen=True)
class CCfgFactors:
    MTM: CCfgFactorMTM | None
    SKEW: CCfgFactorSKEW | None
    RS: CCfgFactorRS | None
    BASIS: CCfgFactorBASIS | None
    TS: CCfgFactorTS | None
    S0BETA: CCfgFactorS0BETA | None
    S1BETA: CCfgFactorS1BETA | None
    CBETA: CCfgFactorCBETA | None
    IBETA: CCfgFactorIBETA | None
    PBETA: CCfgFactorPBETA | None
    CTP: CCfgFactorCTP | None
    CTR: CCfgFactorCTR | None
    CVP: CCfgFactorCVP | None
    CVR: CCfgFactorCVR | None
    CSP: CCfgFactorCSP | None
    CSR: CCfgFactorCSR | None
    NOI: CCfgFactorNOI | None
    NDOI: CCfgFactorNDOI | None
    WNOI: CCfgFactorWNOI | None
    WNDOI: CCfgFactorWNDOI | None
    AMP: CCfgFactorAMP | None
    EXR: CCfgFactorEXR | None
    SMT: CCfgFactorSMT | None
    RWTC: CCfgFactorRWTC | None

    def values(self) -> list[CCfgFactor]:
        res = []
        for _, v in vars(self).items():
            if v is not None:
                res.append(v)
        return res


"""
---------------------------------------------------
Part III: factor class from different configuration
---------------------------------------------------
"""


class CFactorMTM(CFactor):
    def __init__(self, cfg: CCfgFactorMTM, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        major_data = self.load_major(instru)
        adj_major_data = self.truncate(major_data, index="trading_day", before=win_start_date, after=end_date)
        for win, factor_name in zip(self.cfg.wins, self.factor_names):
            adj_major_data[factor_name] = adj_major_data["major_return"].rolling(window=win).sum()
        factor_data = self.get_factor_data(adj_major_data, bgn_date)
        return factor_data


class CFactorSKEW(CFactor):
    def __init__(self, cfg: CCfgFactorSKEW, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        major_data = self.load_major(instru)
        adj_major_data = self.truncate(major_data, index="trading_day", before=win_start_date, after=end_date)
        for win, factor_name in zip(self.cfg.wins, self.factor_names):
            adj_major_data[factor_name] = adj_major_data["major_return"].rolling(window=win).skew()
        factor_data = self.get_factor_data(adj_major_data, bgn_date)
        return factor_data


class CFactorRS(CFactor):
    def __init__(self, cfg: CCfgFactorRS, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        major_data = self.load_major(instru)
        adj_major_data = self.truncate(major_data, index="trading_day", before=bgn_date, after=end_date)
        adj_major_data = adj_major_data.reset_index()
        adj_major_data = adj_major_data[["tp", "trading_day", "ticker"]]

        __min_win = 5
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        preprocess_data = self.load_preprocess(instru=instru)
        adj_data = self.truncate(preprocess_data, index="trading_day", before=win_start_date, after=end_date)
        adj_data["in_stock"] = adj_data["in_stock"].ffill(limit=__min_win).fillna(0)
        for win in self.cfg.wins:
            rspa = f"{self.factor_class}PA{win:03d}"
            rsla = f"{self.factor_class}LA{win:03d}"

            ma = adj_data["in_stock"].rolling(window=win).mean()
            s = adj_data["in_stock"] / ma
            s[s == np.inf] = np.nan  # some maybe resulted from divided by Zero
            adj_data[rspa] = 1 - s

            la = adj_data["in_stock"].shift(win)
            s = adj_data["in_stock"] / la
            s[s == np.inf] = np.nan  # some maybe resulted from divided by Zero
            adj_data[rsla] = 1 - s
        adj_data = adj_data.truncate(before=int(bgn_date)).reset_index()
        adj_data = adj_data[["tp", "trading_day"] + self.factor_names]

        merged_data = pd.merge(left=adj_major_data, right=adj_data, on=["tp", "trading_day"], how="left")
        factor_data = merged_data[["tp", "trading_day", "ticker"] + self.factor_names]
        return factor_data


class CFactorBASIS(CFactor):
    def __init__(self, cfg: CCfgFactorBASIS, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        major_data = self.load_major(instru)
        adj_major_data = self.truncate(major_data, index="trading_day", before=bgn_date, after=end_date)
        adj_major_data = adj_major_data.reset_index()
        adj_major_data = adj_major_data[["tp", "trading_day", "ticker"]]

        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        preprocess_data = self.load_preprocess(instru=instru)
        adj_data = self.truncate(preprocess_data, index="trading_day", before=win_start_date, after=end_date)
        for win in self.cfg.wins:
            f0 = f"{self.factor_class}{win:03d}"
            f1 = f"{self.factor_class}D{win:03d}"
            adj_data[f0] = adj_data["basis_rate"].rolling(window=win, min_periods=int(2 * win / 3)).mean()
            adj_data[f1] = adj_data["basis_rate"] - adj_data[f0]
        adj_data = adj_data.truncate(before=int(bgn_date)).reset_index()
        adj_data = adj_data[["tp", "trading_day"] + self.factor_names]

        merged_data = pd.merge(left=adj_major_data, right=adj_data, on=["tp", "trading_day"], how="left")
        factor_data = merged_data[["tp", "trading_day", "ticker"] + self.factor_names]
        return factor_data


class CFactorTS(CFactor):
    def __init__(self, cfg: CCfgFactorTS, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    @staticmethod
    def cal_roll_return(x: pd.Series, ticker_n: str, ticker_d: str, prc_n: str, prc_d: str):
        if x[ticker_n] == b"" or x[ticker_d] == b"":
            return np.nan
        if x[prc_d] > 0:
            month_d, month_n = int(x[ticker_d][-2:]), int(x[ticker_n][-2:])
            dlt_month = month_d - month_n
            dlt_month = dlt_month + (12 if dlt_month <= 0 else 0)
            return (x[prc_n] / x[prc_d] - 1) / dlt_month * 12 * 100
        else:
            return np.nan

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        major_data = self.load_major(instru)
        adj_major_data = self.truncate(major_data, index="trading_day", before=bgn_date, after=end_date)
        adj_major_data = adj_major_data.reset_index()
        adj_major_data = adj_major_data[["tp", "trading_day", "ticker"]]

        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        preprocess_data = self.load_preprocess(instru=instru)
        adj_data = self.truncate(preprocess_data, index="trading_day", before=win_start_date, after=end_date)
        adj_data["ts"] = adj_data.apply(
            self.cal_roll_return, args=("ticker_major", "ticker_minor", "close_major", "close_minor"), axis=1
        )
        for win in self.cfg.wins:
            f0 = f"{self.factor_class}{win:03d}"
            f1 = f"{self.factor_class}D{win:03d}"
            adj_data[f0] = adj_data["ts"].rolling(window=win, min_periods=int(2 * win / 3)).mean()
            adj_data[f1] = adj_data["ts"] - adj_data[f0]
        adj_data = adj_data.truncate(before=int(bgn_date)).reset_index()
        adj_data = adj_data[["tp", "trading_day"] + self.factor_names]

        merged_data = pd.merge(left=adj_major_data, right=adj_data, on=["tp", "trading_day"], how="left")
        factor_data = merged_data[["tp", "trading_day", "ticker"] + self.factor_names]
        return factor_data


class __CFactorBETA(CFactor):
    @staticmethod
    def merge_xy(x_data: pd.DataFrame, y_data: pd.DataFrame) -> pd.DataFrame:
        adj_data = pd.merge(left=x_data, right=y_data, how="left", left_index=True, right_index=True)
        return adj_data

    def betas_from_wins(self, wins: list[int], input_data: pd.DataFrame, x: str, y: str):
        f0 = f"{self.factor_class}{wins[0]:03d}"
        for i, win in enumerate(wins):
            fi = f"{self.factor_class}{win:03d}"
            input_data[fi] = cal_rolling_beta(df=input_data, x=x, y=y, rolling_window=win)
            if i > 0:
                fid = f"{f0}D{win:03d}"
                input_data[fid] = input_data[f0] - input_data[fi]
        return 0

    def res_from_wins(self, wins: list[int], input_data: pd.DataFrame, x: str, y: str):
        for i, win in enumerate(wins):
            b, fi = f"{self.factor_class}{win:03d}", f"{self.factor_class}{win:03d}RES"
            input_data[fi] = input_data[y] - input_data[x] * input_data[b]
        return 0

    def res_std_from_wins(self, wins: list[int], input_data: pd.DataFrame):
        for i, win in enumerate(wins):
            res, fi = f"{self.factor_class}{win:03d}RES", f"{self.factor_class}{win:03d}RESSTD"
            input_data[fi] = input_data[res].rolling(window=win, min_periods=int(win * 0.6)).std()
        return 0


class CFactorS0BETA(__CFactorBETA):
    def __init__(self, cfg: CCfgFactorS0BETA, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        __x_ret = "NH0100.NHF" if self.universe[instru]["sectorL0"] == "C" else "881001.WI"
        __y_ret = "major_return"
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date)
        adj_market_data = self.get_adj_market_data(win_start_date, end_date)
        adj_data = self.merge_xy(x_data=adj_major_data[["tp", "ticker", __y_ret]], y_data=adj_market_data[[__x_ret]])
        self.betas_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_std_from_wins(self.cfg.wins, adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorS1BETA(__CFactorBETA):
    def __init__(self, cfg: CCfgFactorS1BETA, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        __x_ret, __y_ret = self.universe[instru]["sectorL1"], "major_return"
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date)
        adj_market_data = self.get_adj_market_data(win_start_date, end_date)
        adj_data = self.merge_xy(x_data=adj_major_data[["tp", "ticker", __y_ret]], y_data=adj_market_data[[__x_ret]])
        self.betas_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_std_from_wins(self.cfg.wins, adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorCBETA(__CFactorBETA):
    def __init__(self, cfg: CCfgFactorCBETA, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        __x_ret, __y_ret = "pct_chg", "major_return"
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date, y=__y_ret)
        adj_forex_data = self.get_adj_forex_data(win_start_date, end_date)
        adj_data = self.merge_xy(x_data=adj_major_data[["tp", "ticker", __y_ret]], y_data=adj_forex_data[[__x_ret]])
        self.betas_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_std_from_wins(self.cfg.wins, adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorIBETA(__CFactorBETA):
    def __init__(self, cfg: CCfgFactorIBETA, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        __x_ret, __y_ret = "cpi_rate", "major_return"
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date, y=__y_ret)
        adj_macro_data = self.get_adj_macro_data(win_start_date, end_date)
        adj_data = self.merge_xy(x_data=adj_major_data[["tp", "ticker", __y_ret]], y_data=adj_macro_data[[__x_ret]])
        self.betas_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_std_from_wins(self.cfg.wins, adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorPBETA(__CFactorBETA):
    def __init__(self, cfg: CCfgFactorPBETA, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        __x_ret, __y_ret = "ppi_rate", "major_return"
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date, y=__y_ret)
        adj_macro_data = self.get_adj_macro_data(win_start_date, end_date)
        adj_data = self.merge_xy(x_data=adj_major_data[["tp", "ticker", __y_ret]], y_data=adj_macro_data[[__x_ret]])
        self.betas_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_std_from_wins(self.cfg.wins, adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class __CFactorCXY(CFactor):
    def cal_rolling_top_corr(
        self,
        raw_data: pd.DataFrame,
        bgn_date: np.int32,
        end_date: np.int32,
        x: str,
        y: str,
        wins: list[int],
        tops: list[float],
        sort_var: str,
    ):
        for win, top in ittl.product(wins, tops):
            factor_name = f"{self.factor_class}{win:03d}T{int(top*10):02d}"
            top_size = int(win * top) + 1
            r_data = {}
            for i, trade_date in enumerate(raw_data.index):
                if trade_date < bgn_date:
                    continue
                elif trade_date > end_date:
                    break
                sub_data = raw_data.iloc[i - win + 1 : i + 1]
                r_data[trade_date] = cal_top_corr(sub_data, x=x, y=y, sort_var=sort_var, top_size=top_size)
            raw_data[factor_name] = pd.Series(r_data)
        return 0


class CFactorCTP(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCTP, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date)
        adj_major_data["aver_oi"] = adj_major_data["oi"].rolling(window=2).mean()
        adj_major_data["turnover"] = adj_major_data["vol"] / adj_major_data["aver_oi"]
        x, y = "turnover", "closeM"
        self.cal_rolling_top_corr(
            adj_major_data,
            bgn_date=bgn_date,
            end_date=end_date,
            x=x,
            y=y,
            wins=self.cfg.wins,
            tops=self.cfg.tops,
            sort_var="vol",
        )
        factor_data = self.get_factor_data(adj_major_data, bgn_date=bgn_date)
        return factor_data


class CFactorCTR(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCTR, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date)
        adj_major_data["aver_oi"] = adj_major_data["oi"].rolling(window=2).mean()
        adj_major_data["turnover"] = adj_major_data["vol"] / adj_major_data["aver_oi"]
        x, y = "turnover", "major_return"
        self.cal_rolling_top_corr(
            adj_major_data,
            bgn_date=bgn_date,
            end_date=end_date,
            x=x,
            y=y,
            wins=self.cfg.wins,
            tops=self.cfg.tops,
            sort_var="vol",
        )
        factor_data = self.get_factor_data(adj_major_data, bgn_date=bgn_date)
        return factor_data


class CFactorCVP(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCVP, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date)
        x, y = "vol", "closeM"
        self.cal_rolling_top_corr(
            adj_major_data,
            bgn_date=bgn_date,
            end_date=end_date,
            x=x,
            y=y,
            wins=self.cfg.wins,
            tops=self.cfg.tops,
            sort_var="vol",
        )
        factor_data = self.get_factor_data(adj_major_data, bgn_date=bgn_date)
        return factor_data


class CFactorCVR(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCVR, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date)
        x, y = "vol", "major_return"
        self.cal_rolling_top_corr(
            adj_major_data,
            bgn_date=bgn_date,
            end_date=end_date,
            x=x,
            y=y,
            wins=self.cfg.wins,
            tops=self.cfg.tops,
            sort_var="vol",
        )
        factor_data = self.get_factor_data(adj_major_data, bgn_date=bgn_date)
        return factor_data


class CFactorCSP(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCSP, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        major_data = self.load_major(instru)
        __near_short_term = 20
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.truncate(major_data, index="trading_day", before=win_start_date, after=end_date)
        adj_major_data["sigma"] = adj_major_data["major_return"].fillna(0).rolling(window=__near_short_term).std()
        x, y = "sigma", "closeM"
        self.cal_rolling_top_corr(
            adj_major_data,
            bgn_date=bgn_date,
            end_date=end_date,
            x=x,
            y=y,
            wins=self.cfg.wins,
            tops=self.cfg.tops,
            sort_var="vol",
        )
        factor_data = self.get_factor_data(adj_major_data, bgn_date=bgn_date)
        return factor_data


class CFactorCSR(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCSR, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        major_data = self.load_major(instru)
        __near_short_term = 20
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.truncate(major_data, index="trading_day", before=win_start_date, after=end_date)
        adj_major_data["sigma"] = adj_major_data["major_return"].fillna(0).rolling(window=__near_short_term).std()
        x, y = "sigma", "major_return"
        self.cal_rolling_top_corr(
            adj_major_data,
            bgn_date=bgn_date,
            end_date=end_date,
            x=x,
            y=y,
            wins=self.cfg.wins,
            tops=self.cfg.tops,
            sort_var="vol",
        )
        factor_data = self.get_factor_data(adj_major_data, bgn_date=bgn_date)
        return factor_data


class __CFactorMbrPos(CFactor):
    def __init__(self, cfg: CCfgFactorNOI | CCfgFactorNDOI | CCfgFactorWNOI | CCfgFactorWNDOI, **kwargs):
        if cfg.factor_class not in ["NOI", "NDOI", "WNOI", "WNDOI"]:
            raise ValueError(f"factor class - {cfg.factor_class} is illegal")
        self.cfg = cfg
        self.call_weight_sum = cfg.factor_class in ["WNOI", "WNDOI"]
        self.using_diff = cfg.factor_class in ["NDOI", "WNDOI"]
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_core(self, mbr_pos_data: pd.DataFrame, top: int, instru_oi_data: pd.DataFrame) -> pd.DataFrame:
        def __robust_rate(z: pd.Series) -> float:
            return z.iloc[0] / z.iloc[1] * 100 if z.iloc[1] > 0 else np.nan

        filter_type = mbr_pos_data["info_type"] >= 2  # info_type = 2 or 3
        filter_rank = mbr_pos_data["info_rank"] <= top
        selected_df = mbr_pos_data[filter_type & filter_rank]
        oi_df = pd.pivot_table(
            data=selected_df,
            index="trading_day",
            columns="info_type",
            values="info_qty_dlt" if self.using_diff else "info_qty",
            aggfunc=auto_weight_sum if self.call_weight_sum else "sum",
        )
        oi_df.columns = oi_df.columns.to_flat_index()
        oi_df = oi_df.merge(right=instru_oi_data, left_index=True, right_index=True, how="left")
        oi_df["noi_sum"] = oi_df[2].fillna(0) - oi_df[3].fillna(0)
        oi_df["net"] = oi_df[["noi_sum", "oi_instru"]].apply(__robust_rate, axis=1)
        res = oi_df[["net"]]
        return res

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)

        # load adj major data as header
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date)
        adj_major_data = adj_major_data[["tp", "ticker"]]

        # load preprocess
        adj_preprocess_data = self.get_adj_preprocess_data(instru, win_start_date, end_date)
        adj_preprocess_data = adj_preprocess_data[["oi_instru"]]

        # load member
        adj_mbr_pos_data = self.get_adj_mbr_pos_data(instru, win_start_date, end_date)

        # cal
        res = {}
        for top in self.cfg.tops:
            net_data = self.cal_core(mbr_pos_data=adj_mbr_pos_data, top=top, instru_oi_data=adj_preprocess_data)
            for win in self.cfg.wins:
                mp = int(2 * win / 3)
                factor_name = f"{self.factor_class}{win:03d}T{top:02d}"
                res[factor_name] = net_data["net"].rolling(window=win, min_periods=mp).mean()
        res_df = pd.DataFrame(res)

        # merge to header
        adj_data = pd.merge(
            left=adj_major_data,
            right=res_df,
            left_index=True,
            right_index=True,
            how="left",
        )
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorNOI(__CFactorMbrPos):
    def __init__(self, cfg: CCfgFactorNOI, **kwargs):
        super().__init__(cfg=cfg, **kwargs)


class CFactorNDOI(__CFactorMbrPos):
    def __init__(self, cfg: CCfgFactorNDOI, **kwargs):
        super().__init__(cfg=cfg, **kwargs)


class CFactorWNOI(__CFactorMbrPos):
    def __init__(self, cfg: CCfgFactorWNOI, **kwargs):
        super().__init__(cfg=cfg, **kwargs)


class CFactorWNDOI(__CFactorMbrPos):
    def __init__(self, cfg: CCfgFactorWNDOI, **kwargs):
        super().__init__(cfg=cfg, **kwargs)


class CFactorAMP(CFactor):
    def __init__(self, cfg: CCfgFactorAMP, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    @staticmethod
    def cal_amp(
        sub_data: pd.DataFrame, x: str, sort_var: str, top_size: int, ascending: bool = False
    ) -> tuple[float, float, float]:
        sorted_data = sub_data.sort_values(by=sort_var, ascending=ascending)
        amp_h = sorted_data.head(top_size)[x].mean()
        amp_l = sorted_data.tail(top_size)[x].mean()
        amp_d = amp_h - amp_l
        return amp_h, amp_l, amp_d

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date)
        adj_major_data["amp"] = adj_major_data["high"] / adj_major_data["low"] - 1
        adj_major_data["spot"] = adj_major_data["closeM"]
        for win, lbd in ittl.product(self.cfg.wins, self.cfg.lbds):
            top_size = int(win * lbd) + 1
            factor_h, factor_l, factor_d = [
                f"{self.factor_class}{win:03d}T{int(lbd*10):02d}{_}" for _ in ["H", "L", "D"]
            ]
            r_h_data, r_l_data, r_d_data = {}, {}, {}
            for i, trade_date in enumerate(adj_major_data.index):
                if (trade_date < bgn_date) or (trade_date > end_date):
                    continue
                sub_data = adj_major_data.iloc[i - win + 1 : i + 1]
                rh, rl, rd = self.cal_amp(sub_data=sub_data, x="amp", sort_var="spot", top_size=top_size)
                r_h_data[trade_date], r_l_data[trade_date], r_d_data[trade_date] = rh, rl, rd
            for iter_data, factor in zip([r_h_data, r_l_data, r_d_data], [factor_h, factor_l, factor_d]):
                adj_major_data[factor] = pd.Series(iter_data)
        factor_data = self.get_factor_data(adj_major_data, bgn_date=bgn_date)
        return factor_data


class CFactorEXR(CFactor):
    def __init__(self, cfg: CCfgFactorEXR, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    @staticmethod
    def find_extreme_return(tday_minb_data: pd.DataFrame, ret: str, dfts: list[int]) -> dict[str, float]:
        ret_min, ret_max, ret_median = (
            tday_minb_data[ret].min(),
            tday_minb_data[ret].max(),
            tday_minb_data[ret].median(),
        )
        if (ret_max + ret_min) > (2 * ret_median):
            idx_exr, exr = tday_minb_data[ret].argmax(), -ret_max
        else:
            idx_exr, exr = tday_minb_data[ret].argmin(), -ret_min
        res = {"EXR": exr}
        for d in dfts:
            idx_dxr = idx_exr - d
            dxr = -tday_minb_data[ret].iloc[idx_dxr] if idx_dxr >= 0 else exr
            res[f"DXR{d:02d}"] = dxr
        return res

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date)
        adj_minb_data = self.get_adj_minute_bar_data(instru, win_start_date, end_date, freq=self.cfg.freq)
        adj_minb_data["freq_ret"] = adj_minb_data["close"] / adj_minb_data["preclose"] - 1
        adj_minb_data["freq_ret"] = adj_minb_data["freq_ret"].fillna(0)
        res_srs = adj_minb_data.groupby(by="trading_day").apply(
            self.find_extreme_return, ret="freq_ret", dfts=self.cfg.dfts  # type:ignore
        )
        exr_dxr_df = pd.DataFrame.from_dict(res_srs.to_dict(), orient="index")
        factor_win_dfs: list[pd.DataFrame] = []
        for win in self.cfg.wins:
            rename_mapper = {
                **{"EXR": f"EXR{win:03d}"},
                **{f"DXR{d:02d}": f"DXR{win:03d}D{d:02d}" for d in self.cfg.dfts},
            }
            factor_win_data = exr_dxr_df.rolling(window=win).mean()
            factor_win_data = factor_win_data.rename(mapper=rename_mapper, axis=1)
            for d in self.cfg.dfts:
                exr = f"EXR{win:03d}"
                dxr = f"DXR{win:03d}D{d:02d}"
                axr = f"AXR{win:03d}D{d:02d}"
                factor_win_data[axr] = (factor_win_data[exr] + factor_win_data[dxr] * np.sqrt(2)) * 0.5
            factor_win_dfs.append(factor_win_data)
        concat_factor_data = pd.concat(factor_win_dfs, axis=1, ignore_index=False)
        input_data = pd.merge(
            left=adj_major_data[["tp", "ticker"]],
            right=concat_factor_data,
            left_index=True,
            right_index=True,
            how="left",
        )
        factor_data = self.get_factor_data(input_data, bgn_date=bgn_date)
        return factor_data


class CFactorSMT(CFactor):
    def __init__(self, cfg: CCfgFactorSMT, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    @staticmethod
    def cal_smart_idx(data: pd.DataFrame, ret: str, vol: str) -> pd.Series:
        return data[[ret, vol]].apply(lambda z: np.abs(z[ret]) / np.log(z[vol]) * 1e4 if z[vol] > 1 else 0, axis=1)

    @staticmethod
    def cal_smt(sorted_sub_data: pd.DataFrame, lbd: float, prc: str, ret: str) -> tuple[float, float]:
        # total price and ret
        if (tot_amt_sum := sorted_sub_data["amount"].sum()) > 0:
            tot_w = sorted_sub_data["amount"] / tot_amt_sum
            tot_prc = sorted_sub_data[prc] @ tot_w
            tot_ret = sorted_sub_data[ret] @ tot_w
        else:
            return np.nan, np.nan

        # select smart data from total
        volume_threshold = sorted_sub_data["vol"].sum() * lbd
        n = sum(sorted_sub_data["vol"].cumsum() < volume_threshold) + 1
        smt_df = sorted_sub_data.head(n)

        # smart price and ret
        if (smt_amt_sum := smt_df["amount"].sum()) > 0:
            smt_w = smt_df["amount"] / smt_amt_sum
            smt_prc = smt_df[prc] @ smt_w
            smt_ret = smt_df[ret] @ smt_w
            return (smt_prc / tot_prc - 1) * 1e4, (smt_ret - tot_ret) * 1e4
        else:
            return np.nan, np.nan

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date)

        adj_minb_data = self.get_adj_minute_bar_data(instru, win_start_date, end_date, freq=self.cfg.freq)
        adj_minb_data["freq_ret"] = adj_minb_data["close"] / adj_minb_data["preclose"] - 1
        adj_minb_data["freq_ret"] = adj_minb_data["freq_ret"].fillna(0)

        # contract multiplier is not considered when calculating "vwap"
        # because a price ratio is considered in the final results, not an absolute value of price is considered
        adj_minb_data["vwap"] = (adj_minb_data["amount"] / adj_minb_data["vol"]).ffill()

        # smart idx
        adj_minb_data["smart_idx"] = self.cal_smart_idx(adj_minb_data, ret="freq_ret", vol="vol")

        factor_win_dfs: list[pd.DataFrame] = []
        for win in self.cfg.wins:
            iter_tail_dates = calendar.get_iter_list(bgn_date, end_date)
            base_bgn_date = calendar.get_next_date(iter_tail_dates[0], -win + 1)
            base_end_date = calendar.get_next_date(iter_tail_dates[-1], -win + 1)
            iter_head_dates = calendar.get_iter_list(base_bgn_date, base_end_date)
            p_data, r_data = {}, {}
            for iter_bgn_date, iter_end_date in zip(iter_head_dates, iter_tail_dates):
                sub_data = adj_minb_data.truncate(before=int(iter_bgn_date), after=int(iter_end_date))
                sorted_sub_data = sub_data.sort_values(by="smart_idx", ascending=False)
                p_data[iter_end_date], r_data[iter_end_date] = {}, {}
                for lbd in self.cfg.lbds:
                    p_lbl = f"{self.factor_class}{win:03d}T{int(lbd*10):02d}P"
                    r_lbl = f"{self.factor_class}{win:03d}T{int(lbd*10):02d}R"
                    smt_p, smt_r = self.cal_smt(sorted_sub_data, lbd=lbd, prc="vwap", ret="freq_ret")
                    p_data[iter_end_date][p_lbl], r_data[iter_end_date][r_lbl] = smt_p, smt_r
            factor_win_p_data = pd.DataFrame.from_dict(p_data, orient="index")
            factor_win_r_data = pd.DataFrame.from_dict(r_data, orient="index")
            factor_win_dfs.append(factor_win_p_data)
            factor_win_dfs.append(factor_win_r_data)
        concat_factor_data = pd.concat(factor_win_dfs, axis=1, ignore_index=False)
        input_data = pd.merge(
            left=adj_major_data[["tp", "ticker"]],
            right=concat_factor_data,
            left_index=True,
            right_index=True,
            how="left",
        )
        factor_data = self.get_factor_data(input_data, bgn_date=bgn_date)
        return factor_data


class CFactorRWTC(CFactor):
    def __init__(self, cfg: CCfgFactorRWTC, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    @staticmethod
    def cal_range_weighted_time_center(tday_minb_data: pd.DataFrame, ret: str):
        index_reset_df = tday_minb_data.reset_index()
        pos_idx = index_reset_df[ret] > 0
        neg_idx = index_reset_df[ret] < 0
        pos_grp = index_reset_df.loc[pos_idx, ret]
        neg_grp = index_reset_df.loc[neg_idx, ret]
        pos_wgt = pos_grp.abs() / pos_grp.abs().sum()
        neg_wgt = neg_grp.abs() / neg_grp.abs().sum()
        rwtc_u = pos_grp.index @ pos_wgt / len(tday_minb_data)
        rwtc_d = neg_grp.index @ neg_wgt / len(tday_minb_data)
        rwtc_t = rwtc_u - rwtc_d
        rwtc_v = np.abs(rwtc_t)
        return {"RWTCU": rwtc_u, "RWTCD": rwtc_d, "RWTCT": rwtc_t, "RWTCV": rwtc_v}

    def cal_factor_by_instru(
        self, instru: str, bgn_date: np.int32, end_date: np.int32, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.get_adj_major_data(instru, win_start_date, end_date)

        adj_minb_data = self.get_adj_minute_bar_data(instru, win_start_date, end_date, freq=self.cfg.freq)
        adj_minb_data["freq_ret"] = adj_minb_data["close"] / adj_minb_data["preclose"] - 1
        adj_minb_data["freq_ret"] = adj_minb_data["freq_ret"].fillna(0)
        res_srs = adj_minb_data.groupby(by="trading_day").apply(
            self.cal_range_weighted_time_center, ret="freq_ret"  # type:ignore
        )
        rwtc_df = pd.DataFrame.from_dict(res_srs.to_dict(), orient="index")
        factor_win_dfs: list[pd.DataFrame] = []
        for win in self.cfg.wins:
            rename_mapper = {
                "RWTCU": f"{self.factor_class}{win:03d}U",
                "RWTCD": f"{self.factor_class}{win:03d}D",
                "RWTCT": f"{self.factor_class}{win:03d}T",
                "RWTCV": f"{self.factor_class}{win:03d}V",
            }
            factor_win_data = rwtc_df.rolling(window=win).mean()
            factor_win_data = factor_win_data.rename(mapper=rename_mapper, axis=1)
            factor_win_dfs.append(factor_win_data)
        concat_factor_data = pd.concat(factor_win_dfs, axis=1, ignore_index=False)
        input_data = pd.merge(
            left=adj_major_data[["tp", "ticker"]],
            right=concat_factor_data,
            left_index=True,
            right_index=True,
            how="left",
        )
        factor_data = self.get_factor_data(input_data, bgn_date=bgn_date)
        return factor_data
