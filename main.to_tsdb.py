import argparse
import subprocess as sp
import numpy as np
from rich.progress import Progress, TaskID, track
from hUtils.tools import qtimer
from hUtils.typeDef import TSigArgsTSDB, TFactorComb, TReturnNames, TFactorNames
from hSolutions.sFactorAlg import CCfgFactor

"""
Part I: Shell commands
"""


def gen_shell_cmd(
    src_stack_data_dir: str,
    dst_tsdb_dir: str,
    freq: str,
    prefix: str,
    factor: str,
    fields: list[str] | TReturnNames | TFactorNames,
    tp_end: str,
    key: str,
    ncpu: int,
    precision: int = 32,
) -> str:
    __fields_cmd = " ".join([f"--fields {f}" for f in fields])
    cmd_str = " ".join(
        [
            f"qalign update",
            f"--path {src_stack_data_dir}",
            f"--db {dst_tsdb_dir}",
            f"--tbl {freq}",
            f"{__fields_cmd}",
            f"--end '{tp_end}'",
            f"--prefix {prefix +'.'+ factor}",
            f"--key {key}",
            f"--ncpu {ncpu}",
            f"--float{precision}",
            "-v",
            "--ignore-missing-ticker",
            "--exact",
            "--skip-ii-chk",
        ]
    )
    return cmd_str


def transpose_factor(
    src_stack_data_dir: str,
    dst_tsdb_dir: str,
    freq: str,
    prefix: str,
    factor: str,
    fields: list[str] | TReturnNames | TFactorNames,
    tp_end: str,
    key: str,
    ncpu: int,
):
    cmd_str = gen_shell_cmd(
        src_stack_data_dir=src_stack_data_dir,
        dst_tsdb_dir=dst_tsdb_dir,
        freq=freq,
        prefix=prefix,
        factor=factor,
        fields=fields,
        tp_end=tp_end,
        key=key,
        ncpu=ncpu,
    )
    sp.check_output(args=cmd_str, shell=True)
    return 0


"""
Part II: For factors
"""


@qtimer
def process_by_factor(
    cfg_factor: CCfgFactor,
    src_stack_data_dir: str,
    dst_tsdb_dir: str,
    freq: str,
    prefix: str,
    tp_end: str,
    key: str,
    ncpu: int,
    pb: Progress,
    task_inner: TaskID,
):
    combs: list[TFactorComb] = cfg_factor.get_combs()
    pb.update(task_id=task_inner, completed=0, total=len(combs))
    for factor_class, factor_names, sub_directory in combs:
        transpose_factor(
            src_stack_data_dir=src_stack_data_dir,
            dst_tsdb_dir=dst_tsdb_dir,
            freq=freq,
            prefix=prefix,
            factor=f"{sub_directory}.{factor_class}",
            fields=factor_names,
            tp_end=tp_end,
            key=key,
            ncpu=ncpu,
        )
        pb.update(task_id=task_inner, advance=1)
    return 0


@qtimer
def main_factor(
    cfg_factors: list[CCfgFactor],
    src_stack_data_dir: str,
    dst_tsdb_dir: str,
    freq: str,
    prefix: str,
    tp_end: str,
    key: str,
    ncpu: int,
):
    with Progress() as pb:
        task_outer = pb.add_task("[INF] Processing factor classes", total=len(cfg_factors))
        task_inner = pb.add_task("[INF] Processing factor names", total=0)
        for i, cfg_factor in enumerate(cfg_factors):
            cfg_factor.get_combs()
            process_by_factor(
                cfg_factor=cfg_factor,
                src_stack_data_dir=src_stack_data_dir,
                dst_tsdb_dir=dst_tsdb_dir,
                freq=freq,
                prefix=prefix,
                tp_end=tp_end,
                key=key,
                ncpu=ncpu,
                pb=pb,
                task_inner=task_inner,
            )
            pb.update(task_outer, completed=i + 1)
    return 0


"""
Part III: For test return
"""


def process_test_return(
    lag: int,
    win: int,
    src_stack_data_dir: str,
    dst_tsdb_dir: str,
    freq: str,
    prefix: str,
    tp_end: str,
    key: str,
    ncpu: int,
    pb: Progress,
    task_inner: TaskID,
):
    pb.update(task_id=task_inner, completed=0)
    for save_id in [f"{win:03d}L{lag}", f"{win:03d}L{lag}-NEU"]:
        rets = [f"CloseRtn{save_id}", f"OpenRtn{save_id}"]
        transpose_factor(
            src_stack_data_dir=src_stack_data_dir,
            dst_tsdb_dir=dst_tsdb_dir,
            freq=freq,
            prefix=prefix,
            factor=f"Y.{save_id}",
            fields=rets,
            tp_end=tp_end,
            key=key,
            ncpu=ncpu,
        )
        pb.update(task_id=task_inner, advance=1)
    return 0


def main_test_return(
    lag_and_wins: list[tuple[int, int]],
    src_stack_data_dir: str,
    dst_tsdb_dir: str,
    freq: str,
    prefix: str,
    tp_end: str,
    key: str,
    ncpu: int,
):
    with Progress() as pb:
        task_inner = pb.add_task("Processing sub rets", total=2)
        task_outer = pb.add_task("Processing rets", total=len(lag_and_wins))
        for i, (lag, win) in enumerate(lag_and_wins):
            process_test_return(
                lag=lag,
                win=win,
                src_stack_data_dir=src_stack_data_dir,
                dst_tsdb_dir=dst_tsdb_dir,
                freq=freq,
                prefix=prefix,
                tp_end=tp_end,
                key=key,
                ncpu=ncpu,
                pb=pb,
                task_inner=task_inner,
            )
            pb.update(task_outer, completed=i + 1)
    return 0


"""
Part IV: For signals
"""


def process_signals(
    signal: str,
    fields: TReturnNames,
    src_stack_data_dir: str,
    dst_tsdb_dir: str,
    freq: str,
    prefix: str,
    tp_end: str,
    key: str,
    ncpu: int,
):
    transpose_factor(
        src_stack_data_dir=src_stack_data_dir,
        dst_tsdb_dir=dst_tsdb_dir,
        freq=freq,
        prefix=prefix,
        # factor=f"signals.001L1-NEU.W020-TR01.TW03-Ridge-A00.AGR.M0005",
        # fields=["CloseRtn001L1-NEU"],
        factor=signal,
        fields=fields,
        tp_end=tp_end,
        key=key,
        ncpu=ncpu,
    )
    return 0


def main_signals(
    signals: list[TSigArgsTSDB],
    src_stack_data_dir: str,
    dst_tsdb_dir: str,
    freq: str,
    prefix: str,
    tp_end: str,
    key: str,
    ncpu: int,
):
    for signal, fields in track(signals, description="[INF] Translating signals to tsdb ..."):
        process_signals(
            signal=signal,
            fields=fields,
            src_stack_data_dir=src_stack_data_dir,
            dst_tsdb_dir=dst_tsdb_dir,
            freq=freq,
            prefix=prefix,
            tp_end=tp_end,
            key=key,
            ncpu=ncpu,
        )
    return 0


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To translate data in StackData to TSDB")
    arg_parser.add_argument(
        "--type", type=str, choices=("fac", "ret", "sig"), help="which part to transpose", required=True
    )
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--ncpu", type=int, default=40, help="how many CPUs are used to do the job")
    return arg_parser.parse_args()


if __name__ == "__main__":
    import datetime as dt
    from config import cfg_factors, cfg_path, cfg_strategy

    FREQ = "d01e"
    PREFIX = "team.huxo"
    KEY = "types"
    args = parse_args()
    args_dt_end = dt.datetime.strptime(f"{args.end} 19:00:00.000", "%Y%m%d %H:%M:%S.%f")
    args_tp_end = (args_dt_end + dt.timedelta(days=1)).strftime("%Y%m%d %H:%M:%S.%f")[:-3]

    if args.type == "fac":
        main_factor(
            cfg_factors=list(cfg_factors.values()),
            src_stack_data_dir=cfg_path.path_stack_data_fut,
            dst_tsdb_dir=cfg_path.path_tsdb_futhot,
            freq=FREQ,
            prefix=PREFIX,
            tp_end=args_tp_end,
            key=KEY,
            ncpu=args.ncpu,
        )
    elif args.type == "ret":
        lag_and_wins = list(
            zip(
                [cfg_strategy.CONST["LAG"], 1],
                [cfg_strategy.CONST["WIN"], 1],
            )
        )
        main_test_return(
            lag_and_wins=lag_and_wins,
            src_stack_data_dir=cfg_path.path_stack_data_fut,
            dst_tsdb_dir=cfg_path.path_tsdb_futhot,
            freq=FREQ,
            prefix=PREFIX,
            tp_end=args_tp_end,
            key=KEY,
            ncpu=args.ncpu,
        )
    elif args.type == "sig":
        from config import PATH_CONFIG_MODELS
        from hSolutions.sMclrnManage import load_config_models, get_tests, get_signal_args_tsdb

        config_models = load_config_models(PATH_CONFIG_MODELS)
        tests = get_tests(config_models=config_models)
        signals = get_signal_args_tsdb(tests=tests)
        main_signals(
            signals=signals,
            src_stack_data_dir=cfg_path.path_stack_data_fut,
            dst_tsdb_dir=cfg_path.path_tsdb_futhot,
            freq=FREQ,
            prefix=PREFIX,
            tp_end=args_tp_end,
            key=KEY,
            ncpu=args.ncpu,
        )
    else:
        raise ValueError(f"[ERR] argument type = {args.type} is wrong")
