import numpy as np
import pandas as pd

"""
created @ 2024-01-05
0.  define a class to evaluate the performance of some portfolios
1.  this class provide methods to calculate some frequently used index
"""


class CNAV(object):
    def __init__(
        self,
        input_srs: pd.Series,
        input_type: str,
        annual_factor: float = 250,
        annual_rf_rate: float = 0,
        ret_scale_display: float = 100,
    ):
        """

        :param input_srs:       A. if input_type == "NAV":
                                        the Net-Assets-Value series, with datetime-like index in string format.
                                        The first item in this series COULD NOT be 1, and the class will do the conversion
                                        when initialized. Return will be calculated from nav as following

                                            ret[t] = nav[t]/nav[t-1] -1

                                        and the first item of returns will be filled with 0
                                    elif input_type == "RET":
                                        the Return series, the returns should NOT be multiplied by RETURN_SCALE
                                B. the index of the series is supposed to be continuous, i.e., there are not any missing
                                    dates or timestamp in the index.
        :param input_type:      "NAV" or "RET"
        :param annual_rf_rate:  annualized risk-free rate, must NOT be multiplied by the return scale.
                                the class will do the conversion when initialized
        :param annual_factor:   if the hold period of return is T trading days, then this value should by 250/T
                                |                   |  T | annual factor |               |
                                | daily     returns |  1 |      250      | default value |
                                | weekly    returns |  5 |       50      |               |
                                | monthly   returns | 21 |       12      |               |
                                | quarterly returns | 63 |        4      |               |
        :param ret_scale_display:
        """
        self.return_type = input_type.upper()
        self.annual_factor = annual_factor
        self.annual_rf_rate: float = annual_rf_rate
        self.ret_scale_display: float = ret_scale_display

        if self.return_type == "NAV":
            self.nav_srs: pd.Series = input_srs / input_srs.iloc[0]  # always set the first value to be 1
            self.rtn_srs: pd.Series = (input_srs / input_srs.shift(1) - 1).fillna(0)  # has the same length as nav srs
        elif self.return_type == "RET":
            self.rtn_srs: pd.Series = input_srs
            self.nav_srs: pd.Series = (input_srs + 1).cumprod()
        else:
            raise ValueError(f"input type = {input_type} is illegal")

        self.obs: int = len(input_srs)

        # frequently used performance index
        # primary
        self.return_mean: float = 0
        self.return_std: float = 0
        self.hold_period_return: float = 0
        self.annual_return: float = 0
        self.annual_volatility: float = 0
        self.sharpe_ratio: float = 0
        self.calmar_ratio: float = 0
        self.value_at_risks: dict = {}

        # secondary - A max drawdown scale
        self.max_drawdown_scale: float = 0  # a non-negative float, multiplied by RETURN_SCALE
        self.max_drawdown_scale_idx: str = ""
        self.drawdown_scale_srs: pd.Series = pd.Series(data=0.0, index=self.nav_srs.index)

        # secondary - B max drawdown duration
        self.max_drawdown_duration: int = 0  # a non-negative int, stands for the duration of drawdown
        self.max_drawdown_duration_idx: str = ""
        self.drawdown_duration_srs: pd.Series = pd.Series(data=0, index=self.nav_srs.index)

        # secondary - C max recover duration
        self.max_recover_duration: int = (
            0  # a non-negative int, stands for the first time the nav exceeds the prev high
        )
        self.max_recover_duration_idx: str = ""
        self.recover_duration_srs: pd.Series = pd.Series(data=0, index=self.nav_srs.index)

    def cal_return_mean(self):
        self.return_mean = self.rtn_srs.mean()
        return 0

    def cal_return_std(self):
        self.return_std = self.rtn_srs.std()
        return 0

    def cal_hold_period_return(self):
        self.hold_period_return = self.nav_srs.iloc[-1] - 1
        return 0

    def cal_annual_volatility(self):
        self.annual_volatility = self.rtn_srs.std() * np.sqrt(self.annual_factor)
        return 0

    def cal_annual_return(self, method: str = "linear"):
        if method.lower() == "linear":
            self.annual_return = self.rtn_srs.mean() * self.annual_factor
        elif method.lower() == "compound":
            self.annual_return = np.power(self.nav_srs.iloc[-1], self.annual_factor / self.obs) - 1
        else:
            raise ValueError(f"method = {method} is illegal")
        return 0

    def cal_sharpe_ratio(self):
        diff_srs = self.rtn_srs - self.annual_rf_rate / self.annual_factor
        mu = diff_srs.mean()
        sd = diff_srs.std()
        self.sharpe_ratio = mu / sd * np.sqrt(self.annual_factor)
        return 0

    def cal_max_drawdown_scale(self):
        self.drawdown_scale_srs: pd.Series = 1 - self.nav_srs / self.nav_srs.cummax()
        self.max_drawdown_scale = self.drawdown_scale_srs.max()
        self.max_drawdown_scale_idx = self.drawdown_scale_srs.idxmax()  # type:ignore
        return 0

    def cal_calmar_ratio(self):
        self.cal_annual_return()
        self.cal_max_drawdown_scale()
        self.calmar_ratio = self.annual_return / self.max_drawdown_scale
        return 0

    def cal_max_drawdown_duration(self):
        prev_high = self.nav_srs.iloc[0]
        prev_high_loc = 0
        prev_drawdown_scale = 0.0
        drawdown_loc = 0
        for i, nav_i in enumerate(self.nav_srs):
            if nav_i > prev_high:
                prev_high = nav_i
                prev_high_loc = i
                prev_drawdown_scale = 0
            drawdown_scale = 1 - nav_i / prev_high
            if drawdown_scale > prev_drawdown_scale:
                prev_drawdown_scale = drawdown_scale
                drawdown_loc = i
            self.drawdown_duration_srs.iloc[i] = drawdown_loc - prev_high_loc
        self.max_drawdown_duration = self.drawdown_duration_srs.max()
        self.max_drawdown_duration_idx = self.drawdown_duration_srs.idxmax()  # type:ignore
        return 0

    def cal_max_recover_duration(self):
        prev_high = self.nav_srs.iloc[0]
        prev_high_loc = 0
        for i, nav_i in enumerate(self.nav_srs):
            if nav_i > prev_high:
                self.recover_duration_srs.iloc[i] = 0
                prev_high = nav_i
                prev_high_loc = i
            else:
                self.recover_duration_srs.iloc[i] = i - prev_high_loc
        self.max_recover_duration = self.recover_duration_srs.max()
        self.max_recover_duration_idx = self.recover_duration_srs.idxmax()  # type:ignore
        return

    def cal_value_at_risk(self, qs: tuple[int]):
        self.value_at_risks.update({f"q{q:02d}": np.percentile(self.rtn_srs, q) for q in qs})

    def cal_all_indicators(self, method: str = "linear", qs: tuple = (1, 5, 10, 90, 95, 99)):
        """

        :param method: "linear" or "compound"
        :param qs: Percentage or sequence of percentages for the percentiles to compute.
                     Values must be between 0 and 100 inclusive.
        :return:
        """
        self.cal_return_mean()
        self.cal_return_std()
        self.cal_hold_period_return()
        self.cal_annual_return(method=method)
        self.cal_annual_volatility()
        self.cal_sharpe_ratio()
        self.cal_max_drawdown_scale()
        self.cal_calmar_ratio()
        self.cal_max_drawdown_duration()
        self.cal_max_recover_duration()
        self.cal_value_at_risk(qs=qs)
        return 0

    def to_dict(self, save_type: str = "numeric"):
        """

        :param save_type: "eng": pure English characters, "chs": chinese characters can be read by Latex
        :return:
        """
        if save_type.lower() == "numeric":
            d = {
                "retMean": self.return_mean * self.ret_scale_display,
                "retStd": self.return_std * self.ret_scale_display,
                "hpr": self.hold_period_return * self.ret_scale_display,
                "retAnnual": self.annual_return * self.ret_scale_display,
                "volAnnual": self.annual_volatility * self.ret_scale_display,
                "sharpeRatio": self.sharpe_ratio,
                "calmarRatio": self.calmar_ratio,
                "mdd": self.max_drawdown_scale * self.ret_scale_display,
                "mddT": self.max_drawdown_scale_idx,
                "mddDur": self.max_drawdown_duration,
                "mddDurT": self.max_drawdown_duration_idx,
                "mrd": self.max_recover_duration,
                "mrdT": self.max_recover_duration_idx,
            }
            d.update({k: v for k, v in self.value_at_risks.items()})
        elif save_type.lower() == "string":
            d = {
                "retMean": f"{self.return_mean * self.ret_scale_display:.3f}",
                "retStd": f"{self.return_std * self.ret_scale_display:.3f}",
                "hpr": f"{self.hold_period_return * self.ret_scale_display:.2f}",
                "retAnnual": f"{self.annual_return * self.ret_scale_display:.2f}",
                "volAnnual": f"{self.annual_volatility * self.ret_scale_display:.2f}",
                "sharpeRatio": f"{self.sharpe_ratio:.2f}",
                "calmarRatio": f"{self.calmar_ratio:.2f}",
                "mdd": f"{self.max_drawdown_scale * self.ret_scale_display:.2f}",
                "mddT": f"{self.max_drawdown_scale_idx:s}",
                "mddDur": f"{self.max_drawdown_duration:d}",
                "mddDurT": f"{self.max_drawdown_duration_idx:s}",
                "mrd": f"{self.max_recover_duration:d}",
                "mrdT": f"{self.max_recover_duration_idx:s}",
            }
            d.update({k: f"{v:.3f}" for k, v in self.value_at_risks.items()})
        else:
            raise ValueError(f"save_type = {save_type} is illegal")
        return d

    def display(self):
        print(
            f"| HPR = {self.hold_period_return * self.ret_scale_display:>7.4f} "
            f"| AnnRtn = {self.annual_return * self.ret_scale_display:>7.4f} "
            f"| AnnVol = {self.annual_volatility * self.ret_scale_display:>7.4f} "
            f"| MDD = {self.max_drawdown_scale * self.ret_scale_display:>7.2f} "
            f"| SPR = {self.sharpe_ratio:>7.4f} "
            f"| CMR = {self.calmar_ratio:>7.4f} "
        )
        return 0
