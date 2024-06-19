"""
0. 计算各种因子并对因子进行行业中性化处理
1. 原始因子保存路径
    + 示例 "/var/data/StackData/futures/team/huxo/factors_by_instru/MTM/IC.ss"
    |          tp           | trading_day | ticker | MTM001  | MTM003  | MTM005 | MTM010 | MTM020 | MTM060 | MTM120  | MTM180  | MTM240  |
    |-----------------------|-------------|--------|---------|---------|--------|--------|--------|--------|---------|---------|---------|
    | 20240311 16:00:00.000 |    20240311 | IC2403 |  0.0206 |  0.0214 | 0.0150 | 0.0559 | 0.2148 | 0.0285 | -0.0224 | -0.0530 | -0.0678 |
    | 20240312 16:00:00.000 |    20240312 | IC2403 | -0.0052 |  0.0293 | 0.0177 | 0.0236 | 0.2471 | 0.0194 | -0.0131 | -0.0623 | -0.0781 |
    | 20240313 16:00:00.000 |    20240313 | IC2403 | -0.0020 |  0.0134 | 0.0142 | 0.0467 | 0.1228 | 0.0065 | -0.0142 | -0.0647 | -0.0683 |
    | 20240314 16:00:00.000 |    20240314 | IC2403 | -0.0033 | -0.0105 | 0.0241 | 0.0103 | 0.0532 | 0.0011 | -0.0286 | -0.0764 | -0.0791 |
    | 20240315 16:00:00.000 |    20240315 | IC2406 |  0.0130 |  0.0077 | 0.0230 | 0.0164 | 0.0696 | 0.0242 | -0.0172 | -0.0719 | -0.0643 |
2. 中性化后因子保存路径
    + 示例 "/var/data/StackData/futures/team/huxo/neutral_by_instru/MTM-NEU/IC.ss"
    |          tp           | trading_day | ticker | MTM001-NEU | MTM003-NEU | MTM005-NEU | MTM010-NEU | MTM020-NEU | MTM060-NEU | MTM120-NEU | MTM180-NEU | MTM240-NEU |
    |-----------------------|-------------|--------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
    | 20240311 16:00:00.000 |    20240311 | IC2403 |     0.0046 |     0.0058 |     0.0005 |     0.0134 |     0.0476 |    -0.0053 |     0.0224 |     0.0022 |     0.0169 |
    | 20240312 16:00:00.000 |    20240312 | IC2403 |    -0.0064 |     0.0037 |    -0.0005 |     0.0004 |     0.0455 |    -0.0138 |     0.0164 |    -0.0031 |     0.0077 |
    | 20240313 16:00:00.000 |    20240313 | IC2403 |     0.0005 |    -0.0014 |    -0.0001 |     0.0025 |     0.0062 |    -0.0154 |     0.0144 |    -0.0026 |     0.0083 |
    | 20240314 16:00:00.000 |    20240314 | IC2403 |     0.0003 |    -0.0056 |     0.0045 |    -0.0049 |    -0.0240 |    -0.0142 |     0.0141 |     0.0009 |     0.0074 |
    | 20240315 16:00:00.000 |    20240315 | IC2406 |     0.0039 |     0.0047 |     0.0029 |    -0.0009 |    -0.0041 |    -0.0144 |     0.0166 |     0.0051 |     0.0097 |
"""

import argparse
import numpy as np
from hUtils.calendar import CCalendar
from hUtils.instruments import CInstrumentInfoTable


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To calculate factors")
    arg_parser.add_argument(
        "--factor",
        type=str,
        help="factor class to run",
        required=True,
        choices=("MTM", "SKEW")
        + ("RS", "BASIS", "TS")
        + ("S0BETA", "S1BETA", "CBETA", "IBETA", "PBETA")
        + ("CTP", "CVP", "CSP")
        + ("NOI", "NDOI", "WNOI", "WNDOI")
        + ("AMP", "EXR", "SMT", "RWTC"),
    )
    arg_parser.add_argument("--bgn", type=np.int32, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=np.int32, help="end  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--nomp", default=False, action="store_true", help="not using multiprocess, for debug")
    return arg_parser.parse_args()


if __name__ == "__main__":
    from config import cfg_strategy, cfg_path, cfg_factors
    from hSolutions.sFactor import CFactorNeu

    args = parse_args()
    calendar = CCalendar(cfg_path.path_calendar)
    instru_info_tab = CInstrumentInfoTable(cfg_path.path_instru_info)

    if args.factor == "MTM":
        from hSolutions.sFactorAlg import CFactorMTM

        fac = CFactorMTM(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
        )
    elif args.factor == "SKEW":
        from hSolutions.sFactorAlg import CFactorSKEW

        fac = CFactorSKEW(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
        )
    elif args.factor == "RS":
        from hSolutions.sFactorAlg import CFactorRS

        fac = CFactorRS(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            preprocess_dir=cfg_path.preprocess_dir,
        )
    elif args.factor == "BASIS":
        from hSolutions.sFactorAlg import CFactorBASIS

        fac = CFactorBASIS(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            preprocess_dir=cfg_path.preprocess_dir,
        )
    elif args.factor == "TS":
        from hSolutions.sFactorAlg import CFactorTS

        fac = CFactorTS(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            preprocess_dir=cfg_path.preprocess_dir,
        )
    elif args.factor == "S0BETA":
        from hSolutions.sFactorAlg import CFactorS0BETA

        fac = CFactorS0BETA(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            market_dir=cfg_path.market_dir,
        )
    elif args.factor == "S1BETA":
        from hSolutions.sFactorAlg import CFactorS1BETA

        fac = CFactorS1BETA(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            market_dir=cfg_path.market_dir,
        )
    elif args.factor == "CBETA":
        from hSolutions.sFactorAlg import CFactorCBETA

        fac = CFactorCBETA(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            alternative_dir=cfg_path.alternative_dir,
        )
    elif args.factor == "IBETA":
        from hSolutions.sFactorAlg import CFactorIBETA

        fac = CFactorIBETA(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            alternative_dir=cfg_path.alternative_dir,
        )
    elif args.factor == "PBETA":
        from hSolutions.sFactorAlg import CFactorPBETA

        fac = CFactorPBETA(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            alternative_dir=cfg_path.alternative_dir,
        )
    elif args.factor == "CTP":
        from hSolutions.sFactorAlg import CFactorCTP

        fac = CFactorCTP(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
        )
    elif args.factor == "CVP":
        from hSolutions.sFactorAlg import CFactorCVP

        fac = CFactorCVP(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
        )
    elif args.factor == "CSP":
        from hSolutions.sFactorAlg import CFactorCSP

        fac = CFactorCSP(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
        )
    elif args.factor == "NOI":
        from hSolutions.sFactorAlg import CFactorNOI

        fac = CFactorNOI(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            preprocess_dir=cfg_path.preprocess_dir,
            mbr_pos_dir=cfg_path.mbr_pos_dir,
        )
    elif args.factor == "NDOI":
        from hSolutions.sFactorAlg import CFactorNDOI

        fac = CFactorNDOI(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            preprocess_dir=cfg_path.preprocess_dir,
            mbr_pos_dir=cfg_path.mbr_pos_dir,
        )
    elif args.factor == "WNOI":
        from hSolutions.sFactorAlg import CFactorWNOI

        fac = CFactorWNOI(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            preprocess_dir=cfg_path.preprocess_dir,
            mbr_pos_dir=cfg_path.mbr_pos_dir,
        )
    elif args.factor == "WNDOI":
        from hSolutions.sFactorAlg import CFactorWNDOI

        fac = CFactorWNDOI(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            preprocess_dir=cfg_path.preprocess_dir,
            mbr_pos_dir=cfg_path.mbr_pos_dir,
        )
    elif args.factor == "AMP":
        from hSolutions.sFactorAlg import CFactorAMP

        fac = CFactorAMP(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
        )
    elif args.factor == "EXR":
        from hSolutions.sFactorAlg import CFactorEXR

        fac = CFactorEXR(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            minute_bar_dir=cfg_path.minute_bar_dir,
        )
    elif args.factor == "SMT":
        from hSolutions.sFactorAlg import CFactorSMT

        fac = CFactorSMT(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            minute_bar_dir=cfg_path.minute_bar_dir,
        )
    elif args.factor == "RWTC":
        from hSolutions.sFactorAlg import CFactorRWTC

        fac = CFactorRWTC(
            cfg=cfg_factors[args.factor],
            universe=cfg_strategy.universe,
            factors_by_instru_dir=cfg_path.factors_by_instru_dir,
            major_dir=cfg_path.major_dir,
            minute_bar_dir=cfg_path.minute_bar_dir,
        )
    else:
        raise ValueError(f"factor = {args.factor} is illegal")

    PROCESSES = 12
    fac.main(
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        calendar=calendar,
        call_multiprocess=not args.nomp,
        processes=PROCESSES,
    )

    # Neutralization
    neutralizer = CFactorNeu(
        ref_factor=fac,
        universe=cfg_strategy.universe,
        major_dir=cfg_path.major_dir,
        available_dir=cfg_path.available_dir,
        neutral_by_instru_dir=cfg_path.neutral_by_instru_dir,
    )
    neutralizer.main_neutralize(
        bgn_date=args.bgn,
        end_date=args.end or args.bgn,
        calendar=calendar,
        call_multiprocess=not args.nomp,
        processes=PROCESSES,
    )
