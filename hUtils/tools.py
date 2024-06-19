import os
import shutil
import functools
import datetime as dt
import numpy as np
from itertools import islice


def SetFontColor(c):
    def inner(s: str | int | np.int32 | np.int64 | float | np.float32 | np.float64 | dt.datetime):
        return f"\033[{c}m{s}\033[0m"

    return inner


SFR = SetFontColor(c="0;31;40")  # Red
SFG = SetFontColor(c="0;32;40")  # Green
SFY = SetFontColor(c="0;33;40")  # Yellow
SFB = SetFontColor(c="0;34;40")  # Blue
SFM = SetFontColor(c="0;35;40")  # Magenta
SFC = SetFontColor(c="0;36;40")  # Cyan
SFW = SetFontColor(c="0;37;40")  # White


def check_and_mkdir(dir_path: str, verbose: bool = False):
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass
    if verbose:
        print(f"[INF] Making directory {SFG(dir_path)}")
    return 0


def check_and_makedirs(dir_path: str, verbose: bool = False):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass
    if verbose:
        print(f"[INF] Making directory {SFG(dir_path)}")
    return 0


def remove_files_in_the_dir(dir_path: str):
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
    return 0


def check_and_remove_tree(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    return 0


def qtimer(func):
    # This function shows the execution time of
    # the function object passed
    @functools.wraps(func)  # use this statement to make this function compatible within custom classes
    def wrap_func(*args, **kwargs):
        t1 = dt.datetime.now()
        print(f"[INF] {SFG(t1)} Begin to execute Function {SFG(f'{func.__name__!r}')}")
        result = func(*args, **kwargs)
        t2 = dt.datetime.now()
        duration = (t2 - t1).total_seconds()
        print(f"[INF] {SFG(t2)} Function {SFG(f'{func.__name__!r}')} executed in {SFG(f'{duration:.4f}')} seconds")
        return result

    return wrap_func


def error_handler(error):
    print(f"{SFR('Error')}: {error}", flush=True)


def batched(iterable, batch_size: int):
    i = iter(iterable)
    piece = list(islice(i, batch_size))
    while piece:
        yield piece
        piece = list(islice(i, batch_size))
