import os
import numpy as np
from dataclasses import dataclass
from typing import NewType

TPatchData = NewType("TPatchData", dict[str, dict[tuple[np.int32, bytes], dict[str, float | int]]])
TFactorClass = NewType("TFactorClass", str)
TFactorName = NewType("TFactorName", str)
TFactorNames = NewType("TFactorNames", list[TFactorName])
TFactorClassAndNames = NewType("TFactorClassAndNames", tuple[TFactorClass, TFactorNames])
TFactorComb = NewType("TFactorComb", tuple[TFactorClass, TFactorNames, str])
TFactor = NewType("TFactor", tuple[TFactorClass, TFactorName])
TPrefix = NewType("TPrefix", list[str])
TReturnClass = NewType("TReturnClass", str)
TReturnName = NewType("TReturnName", str)
TReturnNames = NewType("TReturnNames", list[TReturnName])
TSigArgsSS = NewType("TSigArgsSS", tuple[str, str, str])  # (input_dir, output_dir, sid)
TSigArgsTSDB = NewType("TSigArgsTSDB", tuple[TFactorName, TReturnNames])


@dataclass(frozen=True)
class CRet:
    ret_class: TReturnClass
    ret_name: TReturnName
    shift: int

    @property
    def desc(self) -> str:
        return f"{self.ret_class}.{self.ret_name}"


@dataclass(frozen=True)
class CSlcFacs:
    top_ratio: float

    @property
    def desc(self) -> str:
        return f"TR{int(self.top_ratio*100):02d}"


@dataclass(frozen=True)
class CModel:
    model_type: str
    model_args: dict

    @property
    def desc(self) -> str:
        return f"{self.model_type}"


@dataclass(frozen=True)
class CTestFtSlc:
    trn_win: int  # one of [60, 120, 240]
    sector: str
    ret: CRet
    facs_pool: list[TFactor]


@dataclass(frozen=True)
class CTest:
    unique_Id: str
    trn_win: int  # one of [60, 120, 240]
    sector: str
    ret: CRet
    facs: CSlcFacs
    model: CModel

    @property
    def tw(self) -> str:
        return f"W{self.trn_win:03d}"

    @property
    def layers(self) -> list[str]:
        return [
            self.ret.ret_class,  # 001L1-NEU
            self.tw,  # W060
            self.facs.desc,  # TR01
            self.model.desc,  # Ridge
            self.sector,  # AGR
            self.unique_Id,  # M0005
            self.ret.ret_name,  # CloseRtn001L1
        ]

    @property
    def prefix(self) -> list[str]:
        return self.layers[:-1]

    @property
    def save_tag_mdl(self) -> str:
        return ".".join(self.layers)

    @property
    def save_tag_prd(self) -> str:
        return os.path.join(*self.prefix)
