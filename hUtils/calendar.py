import datetime as dt
import numpy as np
import pandas as pd


class CCalendar(object):
    def __init__(self, calendar_path: str):
        calendar_df = pd.read_csv(calendar_path, dtype=np.int32, header=None, names=["trading_day"])
        self.__trade_dates: list[np.int32] = calendar_df["trading_day"].tolist()

    @property
    def last_date(self) -> np.int32:
        return self.__trade_dates[-1]

    @property
    def first_date(self) -> np.int32:
        return self.__trade_dates[0]

    @property
    def trade_dates(self) -> list[np.int32]:
        return self.__trade_dates

    def get_iter_list(self, bgn_date: np.int32, end_date: np.int32, ascending: bool = True) -> list[np.int32]:
        """

        :param bgn_date: both trading day and non-trading day are OK
        :param end_date: both trading day and non-trading day are OK
        : return:
        """
        res = []
        for t_date in self.__trade_dates:
            if t_date < bgn_date:
                continue
            if t_date > end_date:
                break
            res.append(t_date)
        return res if ascending else sorted(res, reverse=True)

    def shift_iter_dates(self, iter_dates: list[np.int32], shift: int) -> list[np.int32]:
        """

        :param iter_dates:
        :param shift: > 0, in the future
                      < 0, in the past
        :return:
        """
        if shift >= 0:
            new_dates = [self.get_next_date(iter_dates[-1], shift=s) for s in range(1, shift + 1)]
            shift_dates = iter_dates[shift:] + new_dates
        else:  # shift < 0
            new_dates = [self.get_next_date(iter_dates[0], shift=s) for s in range(shift, 0)]
            shift_dates = new_dates + iter_dates[:shift]
        return shift_dates

    @staticmethod
    def convert_date_to_tp(trading_day: np.int32, timestamp: str):
        """
        :param timestamp: format = "HH:MM:SS", like "16:00:00"
        :return:
        """
        return dt.datetime.strptime(f"{trading_day} {timestamp}", "%Y%m%d %H:%M:%S").timestamp() * 10**9

    def get_dates_header(self, bgn_date: np.int32, end_date: np.int32, timestamp: str) -> pd.DataFrame:
        """
        :param timestamp: format = "HH:MM:SS", like "16:00:00"
        :return:
        """

        h = pd.DataFrame({"trading_day": self.get_iter_list(bgn_date, end_date)})
        h["tp"] = h["trading_day"].map(lambda z: self.convert_date_to_tp(z, timestamp))
        return h

    def get_sn(self, base_date: np.int32) -> int:
        return self.__trade_dates.index(base_date)

    def get_date(self, sn: int) -> np.int32:
        return self.__trade_dates[sn]

    def get_next_date(self, this_date: np.int32, shift: int = 1) -> np.int32:
        """

        :param this_date:
        :param shift: > 0, get date in the future; < 0, get date in the past
        :return:
        """

        this_sn = self.get_sn(this_date)
        next_sn = this_sn + shift
        return self.__trade_dates[next_sn]

    def get_start_date(self, bgn_date: np.int32, max_win: int, shift: int) -> np.int32:
        return self.get_next_date(bgn_date, -max_win + shift)

    def get_last_day_of_month(self, month: np.int32):
        """
        :param month: like 202403

        """

        threshold = month * 100 + 31
        for t in self.__trade_dates[::-1]:
            if t <= threshold:
                return t
        raise ValueError(f"Could not find last day for {month}")

    def get_first_day_of_month(self, month: np.int32):
        """
        :param month: like 202403

        """

        threshold = month * 100 + 1
        for t in self.__trade_dates:
            if t >= threshold:
                return t
        raise ValueError(f"Could not find first day for {month}")

    def get_last_days_in_range(self, bgn_date: np.int32, end_date: np.int32) -> list[np.int32]:
        res = []
        for this_day, next_day in zip(self.__trade_dates[:-1], self.__trade_dates[1:]):
            if this_day < bgn_date:
                continue
            elif this_day > end_date:
                break
            else:
                if this_day // 100 != next_day // 100:
                    res.append(this_day)
        return res

    def split_by_month(self, dates: list[np.int32]) -> dict[np.int32, list[np.int32]]:
        res = {}
        if dates:
            for t in dates:
                m = t // 100
                if m not in res:
                    res[m] = [t]
                else:
                    res[m].append(t)
        return res

    @staticmethod
    def move_date(trade_date: np.int32, move_days: int = 1) -> np.int32:
        """

        :param trade_date:
        :param move_days: >0, to the future; <0, to the past
        :return:
        """
        s, d = str(trade_date), dt.timedelta(days=move_days)
        n = dt.datetime.strptime(s, "%Y%m%d") + d
        return np.int32(n.strftime("%Y%m%d"))

    @staticmethod
    def get_next_month(month: np.int32, s: int) -> np.int32:
        """

        :param month: format = YYYYMM
        :param s: > 0 in the future
                  < 0 in the past
        :return:
        """
        y, m = month // 100, month % 100
        dy, dm = s // 12, s % 12
        ny, nm = y + dy, m + dm
        if nm > 12:
            ny, nm = ny + 1, nm - 12
        return ny * 100 + nm
