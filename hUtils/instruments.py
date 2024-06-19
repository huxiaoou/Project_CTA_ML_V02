import re
import numpy as np
import pandas as pd
from dataclasses import dataclass


def parse_instrument_from_contract(contract_id: str) -> str:
    return re.sub(pattern="[0-9]", repl="", string=contract_id)


class CInstrumentInfoTable(object):
    def __init__(self, instru_info_path: str, index_label: str = "instrumentId"):
        """

        :param instru_info_path: InstrumentInfo file path, could be a txt(csv) or xlsx
        :param index_label: "instrumentId"(like "a.dce") or "windCode"(like "A.DCE")
        """

        self.instrument_info_df = pd.read_csv(instru_info_path).set_index(index_label)
        self.instrument_info_df["precision"] = self.instrument_info_df["miniSpread"].map(
            lambda z: max(int(-np.floor(np.log10(z))), 0)
        )

    def get_universe(self, code_format: str = "wind") -> list[str]:
        """

        :param code_format: one of ["wind", "vanilla"]
        :return:
        """
        if code_format.upper() == "WIND":
            try:
                return self.instrument_info_df["windCode"].tolist()
            except KeyError:
                return self.instrument_info_df.index.tolist()
        elif code_format.upper() == "VANILLA":
            try:
                return self.instrument_info_df["instrumentId"].tolist()
            except KeyError:
                return self.instrument_info_df.index.tolist()
        else:
            print(f"... {code_format} not a legal input for argument 'code_format', please check again")
            raise ValueError

    def get_multiplier(self, instrument_id: str):
        return self.instrument_info_df.at[instrument_id, "contractMultiplier"]

    def get_multiplier_from_contract(self, contract: str):
        instrument_id = parse_instrument_from_contract(contract)
        return self.get_multiplier(instrument_id)

    def get_mini_spread(self, instrument_id: str):
        return self.instrument_info_df.at[instrument_id, "miniSpread"]

    def get_precision(self, instrument_id: str):
        return self.instrument_info_df.at[instrument_id, "precision"]

    def get_exchange_id(self, instrument_id: str):
        return self.instrument_info_df.at[instrument_id, "exchangeId"]

    def get_exchange_id_eng(self, instrument_id: str):
        exchange_id_full = self.get_exchange_id(instrument_id)
        exchange_id_eng = {
            "DCE": "DCE",
            "CZCE": "CZC",
            "SHFE": "SHF",
            "INE": "INE",
            "GFE": "GFE",
            "CFFEX": "CFE",
        }[exchange_id_full]
        return exchange_id_eng

    def get_exchange_id_chs(self, instrument_id: str):
        exchange_id_full = self.get_exchange_id(instrument_id)
        exchange_id_chs = {
            "DCE": "大商所",
            "CZCE": "郑商所",
            "SHFE": "上期所",
            "INE": "上海能源",
            "GFE": "广期所",
            "CFFEX": "中金所",
        }[exchange_id_full]
        return exchange_id_chs

    def get_wind_code(self, instrument_id: str):
        return self.instrument_info_df.at[instrument_id, "windCode"]

    def get_ngt_sec_end_hour(self, instrument_id: str):
        return self.instrument_info_df.at[instrument_id, "ngtSecEndHour"]

    def get_ngt_sec_end_minute(self, instrument_id: str):
        return self.instrument_info_df.at[instrument_id, "ngtSecEndMinute"]

    def convert_contract_from_vanilla_to_wind(self, contract_id: str) -> str:
        """

        :param contract_id: general contract id, such as "j2209"
        :return: "J2209.DCE"
        """

        _instrument_id = parse_instrument_from_contract(contract_id=contract_id)
        _exchange_id = self.get_wind_code(instrument_id=_instrument_id).split(".")[1]
        return contract_id.upper() + "." + _exchange_id[0:3]

    @staticmethod
    def convert_contract_from_wind_to_vanilla(contract_id: str) -> str:
        """

        :param contract_id: general contract id, such as "J2209.DCE"
        :return: "j2209"
        """
        cid, exchange = contract_id.split(".")
        return cid if exchange in ["CFE", "CZC"] else cid.lower()

    @staticmethod
    def fix_contract_id_alt(ticker: str, exchange: str, instru: str, trade_date: str) -> str:
        """

        :param ticker: it should have a format like "MA005", in which "05" = May
                       however the first "1" is ambiguous, since it could be either
                       "201005" or "202005"
                       this function is designed to solve this problem
        :param exchange_id: CZC, DCE, SHF, INE.
        :param instru: "MA"
        :param trade_date: on which day, this contract is traded
        :return:
        """
        len_ticker, len_instru = len(ticker), len(instru)  # len("MA105"), len("MA")
        if exchange not in ["CZC", "CZCE"]:  # this problem only happens for CZC
            return ticker
        if len_ticker - len_instru > 3:  # some old contract do have format like "MA1105"
            return ticker  # in this case Nothing should be done
        td = int(trade_date[2])  # decimal year to be inserted, "X" in "20XYMMDD", "1" in "20[1]0 05 01"
        ty = int(trade_date[3])  # trade year number,           "Y" in "20XYMMDD", "0" in "201[0] 05 01"
        cy = int(ticker[len_instru])  # contract year, "1" in "MA105"
        if cy < ty:
            # contract year should always be greater than or equal to the trade year
            # if not, decimal year +=1
            td += 1
        return ticker[0:len_instru] + str(td) + ticker[len_instru:]

    def fix_contract_id(self, ticker: str, trade_date: str) -> str:
        """

        :param ticker: it should have a format like "MA105", in which "05" = May
                         however "1" is ambiguous, since it could be either "2011" or "2021"
                         this function is designed to solve this problem
        :param trade_date: on which day, this contract is traded
        :return:
        """

        instru = parse_instrument_from_contract(ticker)  # "MA"
        exchange = self.get_exchange_id(instru)  # "CZCE"
        return self.fix_contract_id_alt(ticker, exchange, instru, trade_date)


@dataclass(frozen=True)
class CContract(object):
    contract: str
    instrument: str
    exchange: str
    contract_multiplier: int

    def contract_and_instru_id(self) -> tuple[str, str]:
        return self.contract, self.instrument

    @staticmethod
    def gen_from_contract_id(contract: str, instru_info_tab: CInstrumentInfoTable) -> "CContract":
        instrument = parse_instrument_from_contract(contract)
        return CContract(
            contract=contract,
            instrument=instrument,
            exchange=instru_info_tab.get_exchange_id(instrument),
            contract_multiplier=instru_info_tab.get_multiplier(instrument),
        )

    @staticmethod
    def gen_from_other(contract: str, other: "CContract") -> "CContract":
        if isinstance(other, CContract):
            return CContract(
                contract=contract,
                instrument=other.instrument,
                exchange=other.exchange,
                contract_multiplier=other.contract_multiplier,
            )
        else:
            print(f"{other} is not an instance of CContract")
            raise TypeError
