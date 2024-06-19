"""
update @ 20240521 15:06:00

从Stack数据中插入到TSDB数据时的bug, 主力合约换月时插入失败。

下面的例子中20240318有原始的TSDB数据，以及Stack数据，但插入后没有数据

原始TSDB数据

python ~/UtilityTools/view_tsdb.py --tsdb /var/TSDB/FutHot --freq d01e --table . --values open,close --bgn 20240312 --end 20240322 --ticker 'al'

ViewTSDB
------------------------------------------------------------------------------------------------------------------------
tsdb     =                                                                           /var/TSDB/FutHot
freq     =                                                                                       d01e
table    =                                                                                          .
values   =                                                                          ['open', 'close']
begin tp =                                                                      20240312 08:00:00.000
end   tp =                                                                      20240322 17:00:00.000
max rows =                                                                                         12
dropna   =                                                                                      False
ticker   =                                                                                         al
tp       =                                                                                          0
sort     =
------------------------------------------------------------------------------------------------------------------------
                      tp  ii     ticker  trading_day      date       time       open      close
0    1710230400000000000   0  b'al2404'     20240312  20240312  160000000 19200.0000 19180.0000
71   1710316800000000000   0  b'al2404'     20240313  20240313  160000000 19205.0000 19235.0000
142  1710403200000000000   0  b'al2404'     20240314  20240314  160000000 19270.0000 19150.0000
213  1710489600000000000   0  b'al2404'     20240315  20240315  160000000 19150.0000 19295.0000
284  1710748800000000000   0  b'al2404'     20240318  20240318  160000000 19280.0000 19275.0000 ***
355  1710835200000000000   0  b'al2405'     20240319  20240319  160000000 19290.0000 19215.0000
426  1710921600000000000   0  b'al2405'     20240320  20240320  160000000 19165.0000 19285.0000
497  1711008000000000000   0  b'al2405'     20240321  20240321  160000000 19320.0000 19485.0000
568  1711094400000000000   0  b'al2405'     20240322  20240322  160000000 19510.0000 19400.0000
-----------------------------------------------------------------------------------------------------------------------

原始Stack数据

python ~/UtilityTools/view_ss.py --file /var/data/StackData/futures/team/huxo/factors_by_instru/SBETA/al.ss --parsetp --bgn 20240312
ViewSS
                      tp  trading_day     ticker  SBETA010  SBETA020  SBETA060  SBETA120  SBETA180  SBETA240
2959 2024-03-12 16:00:00     20240312  b'al2404'  0.291790  0.396789  0.581434  0.457691  0.507519  0.530919
2960 2024-03-13 16:00:00     20240313  b'al2404'  0.249088  0.417776  0.573818  0.446713  0.506096  0.530284
2961 2024-03-14 16:00:00     20240314  b'al2404'  0.161776  0.433516  0.570832  0.438590  0.507132  0.531847
2962 2024-03-15 16:00:00     20240315  b'al2404'  0.309214  0.494071  0.568048  0.449667  0.503443  0.530174
2963 2024-03-18 16:00:00     20240318  b'al2404'  0.351023  0.452964  0.531004  0.445648  0.496004  0.529336 ***
2964 2024-03-19 16:00:00     20240319  b'al2405'  0.385005  0.438468  0.527609  0.438219  0.495599  0.530775
2965 2024-03-20 16:00:00     20240320  b'al2405'  0.315867  0.303626  0.508354  0.434922  0.492411  0.532021
2966 2024-03-21 16:00:00     20240321  b'al2405'  0.407848  0.394491  0.530000  0.447005  0.489856  0.543426
2967 2024-03-22 16:00:00     20240322  b'al2405'  0.437821  0.426550  0.519118  0.449769  0.488663  0.542481

插入到TSDB后的数据

python ~/UtilityTools/view_tsdb.py --tsdb /var/TSDB/FutHot --freq d01e --table team.huxo.factors_by_instru.SBETA --values SBETA010 --bgn 20240312 --end 20240322 --ticker 'al'
ViewTSDB
------------------------------------------------------------------------------------------------------------------------
tsdb     =                                                                           /var/TSDB/FutHot
freq     =                                                                                       d01e
table    =                                                          team.huxo.factors_by_instru.SBETA
values   =                                                                               ['SBETA010']
begin tp =                                                                      20240312 08:00:00.000
end   tp =                                                                      20240322 17:00:00.000
max rows =                                                                                         12
dropna   =                                                                                      False
ticker   =                                                                                         al
tp       =                                                                                          0
sort     =
------------------------------------------------------------------------------------------------------------------------
                      tp  ii     ticker  trading_day      date       time  team.huxo.factors_by_instru.SBETA.SBETA010
0    1710230400000000000   0  b'al2404'     20240312  20240312  160000000                                      0.2918
71   1710316800000000000   0  b'al2404'     20240313  20240313  160000000                                      0.2491
142  1710403200000000000   0  b'al2404'     20240314  20240314  160000000                                      0.1618
213  1710489600000000000   0  b'al2404'     20240315  20240315  160000000                                      0.3092
284  1710748800000000000   0  b'al2404'     20240318  20240318  160000000                                         NaN *** 数据缺失
355  1710835200000000000   0  b'al2405'     20240319  20240319  160000000                                      0.3850
426  1710921600000000000   0  b'al2405'     20240320  20240320  160000000                                      0.3159
497  1711008000000000000   0  b'al2405'     20240321  20240321  160000000                                      0.4078
568  1711094400000000000   0  b'al2405'     20240322  20240322  160000000                                      0.4378
------------------------------------------------------------------------------------------------------------------------


"""

import argparse
import os
import numpy as np
import pandas as pd
from rich.progress import track
from hUtils.tools import SFR, SFG, SFY, qtimer
from hUtils.ioPlus import PySharedStackPlus, CTsdbPlus
from hSolutions.sFactorAlg import CCfgFactor


def load_stack_data(
    stack_root_dir: str,
    prefix: str,
    factor_class: str,
    bgn_date: np.int32,
    end_date: np.int32,
) -> pd.DataFrame:
    ss_file_dir = os.path.join(stack_root_dir, prefix, factor_class)
    dfs: list[pd.DataFrame] = []
    for instru_file in os.listdir(ss_file_dir):
        instru_path = os.path.join(ss_file_dir, instru_file)
        ss = PySharedStackPlus(instru_path)
        df = pd.DataFrame(ss.read_all())
        df = df.set_index("trading_day").truncate(before=int(bgn_date), after=int(end_date)).reset_index()
        dfs.append(df)
    stack_data = pd.concat(dfs, axis=0, ignore_index=True)
    return stack_data


def reformat_stack_data(stack_data: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    rft_stack_data = stack_data.sort_values(by=["tp", "trading_day", "ticker"])
    rft_stack_data = rft_stack_data.dropna(axis=0, how="all", subset=factor_names)
    rft_stack_data = rft_stack_data[rft_stack_data["ticker"] != b""]
    rft_stack_data = rft_stack_data[["tp", "trading_day", "ticker"] + factor_names]
    rft_stack_data = rft_stack_data.set_index(["tp", "trading_day", "ticker"])
    return rft_stack_data


def load_tsdb(
    tsdb_root_dir: str,
    freq: str,
    value_columns: list[str],
    bgn_date: np.int32,
    end_date: np.int32,
) -> pd.DataFrame:
    db_reader = CTsdbPlus(tsdb_root_dir)
    tsdb_data = db_reader.read_range(
        value_columns=value_columns,
        beg_date=str(bgn_date),
        end_date=str(end_date),
        freq=freq.replace("/", ""),
    )
    return tsdb_data


def update_ticker_from_uid(d01e_raw_data: pd.DataFrame, d01b_raw_data: pd.DataFrame) -> pd.DataFrame:
    new_data = pd.merge(
        left=d01e_raw_data,
        right=d01b_raw_data[["ticker", "trading_day", "uid"]],
        on=["ticker", "trading_day"],
        how="left",
    )
    se = len(d01e_raw_data)
    sb = len(d01b_raw_data)
    sn = len(new_data)
    if se == sb == sn:
        # to check ticker = "SR201" contains both uid = "SR1201" and uid = "SR2201"
        # using the following code
        """
        for ticker, ticker_df in new_data.groupby(by="ticker"):
            if ticker == b"SR201":
                print(ticker_df)
        """

        return new_data
    else:
        print(f"[INF] Size of d01e data = {se}")
        print(f"[INF] Size of d01b data = {sb}")
        print(f"[INF] Size of new  data = {sn}")
        raise ValueError(f"[ERR] Size of d01b <> Size of d01e")


def reformat_tsdb_data(tsdb_data: pd.DataFrame, value_columns: list[str], factor_names: list[str]) -> pd.DataFrame:
    rename_mapper = {k: v for k, v in zip(value_columns, factor_names)}
    rft_tsdb_data = tsdb_data.rename(mapper=rename_mapper, axis=1)
    rft_tsdb_data["ticker"] = rft_tsdb_data["uid"]
    rft_tsdb_data = rft_tsdb_data.sort_values(by=["tp", "trading_day", "ticker"])
    rft_tsdb_data = rft_tsdb_data.dropna(axis=0, how="all", subset=factor_names)
    rft_tsdb_data = rft_tsdb_data[["tp", "trading_day", "ticker"] + factor_names]
    rft_tsdb_data = rft_tsdb_data.set_index(["tp", "trading_day", "ticker"])
    return rft_tsdb_data


def diff_between_tsdb_and_stack(
    factor_class: str,
    rft_tsdb_data: pd.DataFrame,
    rft_stack_data: pd.DataFrame,
    bgn_date: np.int32,
    end_date: np.int32,
    eps: float,
):
    if (sp0 := rft_tsdb_data.shape) != (sp1 := rft_stack_data.shape):
        print(f"[INF] shape of tsdb  = {sp0}")
        print(f"[INF] shape of stack = {sp1}")
        print(f"[{SFR('ERR')}] Data not aligned for {SFR(factor_class)}")
    else:
        print(f"[INF] {SFG(factor_class)}: {SFG(len(rft_tsdb_data))} items are checked")
        diff_data = rft_tsdb_data.fillna(0) - rft_stack_data.fillna(0)
        filter_error = diff_data.abs().mean(axis=1) > eps
        diff_data = diff_data[filter_error]
        if not diff_data.empty:
            print(f"[{SFR('ERR')}] The following {SFY(len(diff_data))} differences are found for {SFR(factor_class)}")
            print(diff_data)
        else:
            print(f"[INF] {SFG(factor_class)}: No errors are found from {SFG(int(bgn_date))} to {SFG(int(end_date))}")
    return 0


def check_factor_class(
    factor_class: str,
    factor_names: list[str],
    bgn_date: np.int32,
    end_date: np.int32,
    stack_root_dir: str,
    tsdb_root_dir: str,
    freq: str,
    prefix: list[str],
    eps: float,
):
    # --- load stack
    stack_data = load_stack_data(stack_root_dir, os.path.sep.join(prefix), factor_class, bgn_date, end_date)
    rft_stack_data = reformat_stack_data(stack_data, factor_names=factor_names)

    # --- load tsdb
    value_columns = [f"{'.'.join(prefix)}.{factor_class}.{_}" for _ in factor_names]
    d01e_raw_data = load_tsdb(tsdb_root_dir, freq, value_columns, bgn_date, end_date)
    d01b_raw_data = load_tsdb(tsdb_root_dir, "d01b", ["uid"], bgn_date, end_date)
    tsdb_data = update_ticker_from_uid(d01e_raw_data=d01e_raw_data, d01b_raw_data=d01b_raw_data)
    rft_tsdb_data = reformat_tsdb_data(tsdb_data, value_columns, factor_names=factor_names)

    # --- check
    diff_between_tsdb_and_stack(
        factor_class=factor_class,
        rft_tsdb_data=rft_tsdb_data,
        rft_stack_data=rft_stack_data,
        bgn_date=bgn_date,
        end_date=end_date,
        eps=eps,
    )
    return 0


@qtimer
def main_check(
    cfg_factor: CCfgFactor,
    diff_ns: list[int],
    bgn_date: np.int32,
    end_date: np.int32,
    stack_root_dir: str,
    tsdb_root_dir: str,
    freq: str,
    prefix_user: list[str],
    eps: float,
):
    combs = cfg_factor.get_combs(diff_ns=diff_ns)
    for factor_class, factor_names, sub_directory in track(combs, description="[INF] Check for factors"):
        # for factor_class, factor_names, sub_directory in combs:
        check_factor_class(
            factor_class=factor_class,
            factor_names=factor_names,
            bgn_date=bgn_date,
            end_date=end_date,
            stack_root_dir=stack_root_dir,
            tsdb_root_dir=tsdb_root_dir,
            freq=freq,
            prefix=prefix_user + [sub_directory],
            eps=eps,
        )
    return 0


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To calculate major return and some other data")
    arg_parser.add_argument(
        "--factor",
        type=str,
        help="factor class to run",
        required=True,
        choices=("MTM", "VOL", "RVOL", "SKEW", "CV")
        + ("CTP", "CVP", "CSP", "VAL", "HR")
        + ("SR", "SIZE", "LIQUID", "RS", "BASIS")
        + ("TS", "NOI", "NDOI", "WNOI", "WNDOI")
        + ("CBETA", "IBETA", "MBETA", "SBETA"),
    )
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--eps", type=float, default=1e-4, help="epsilon to control error level, default is 1e-6")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from config import cfg_strategy, cfg_factors, cfg_path

    PREFIX_USER = ["team", "huxo"]
    FREQ = "d01e"  # Only this frequency has been tested

    args = parse_args()
    diff_ns = cfg_strategy.factors_diff["n"]

    cfg_factor: CCfgFactor = cfg_factors[args.factor]
    main_check(
        cfg_factor=cfg_factor,
        diff_ns=diff_ns,
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        stack_root_dir=cfg_path.path_stack_data_fut,
        tsdb_root_dir=cfg_path.path_tsdb_futhot,
        freq=FREQ,
        prefix_user=PREFIX_USER,
        eps=args.eps,
    )
