bgn_date="20120104"
bgn_date_ml="20170203"  # machine learning bgn date
bgn_date_sig="20170703" # signal bgn date
bgn_date_sim="20180102" # simulation bgn date
end_date="20240131"     # TSDB Futhot may wrong after this date

# factors
rm -r /var/data/StackData/futures/team/huxo/factors_by_instru/*
rm -r /var/data/StackData/futures/team/huxo/neutral_by_instru/*
rm -r /var/TSDB/FutHot/d01/e/team/huxo/factors_by_instru/*
rm -r /var/TSDB/FutHot/d01/e/team/huxo/neutral_by_instru/*
python main.factors.py --bgn $bgn_date --end $end_date --factor MTM
python main.factors.py --bgn $bgn_date --end $end_date --factor SKEW
python main.factors.py --bgn $bgn_date --end $end_date --factor RS
python main.factors.py --bgn $bgn_date --end $end_date --factor BASIS
python main.factors.py --bgn $bgn_date --end $end_date --factor TS
python main.factors.py --bgn $bgn_date --end $end_date --factor S0BETA
python main.factors.py --bgn $bgn_date --end $end_date --factor S1BETA
python main.factors.py --bgn $bgn_date --end $end_date --factor CBETA
python main.factors.py --bgn $bgn_date --end $end_date --factor IBETA
python main.factors.py --bgn $bgn_date --end $end_date --factor PBETA
python main.factors.py --bgn $bgn_date --end $end_date --factor CTP
python main.factors.py --bgn $bgn_date --end $end_date --factor CTR
python main.factors.py --bgn $bgn_date --end $end_date --factor CVP
python main.factors.py --bgn $bgn_date --end $end_date --factor CVR
python main.factors.py --bgn $bgn_date --end $end_date --factor CSP
python main.factors.py --bgn $bgn_date --end $end_date --factor CSR
python main.factors.py --bgn $bgn_date --end $end_date --factor NOI
python main.factors.py --bgn $bgn_date --end $end_date --factor NDOI
python main.factors.py --bgn $bgn_date --end $end_date --factor WNOI
python main.factors.py --bgn $bgn_date --end $end_date --factor WNDOI
python main.factors.py --bgn $bgn_date --end $end_date --factor AMP
python main.factors.py --bgn $bgn_date --end $end_date --factor EXR
python main.factors.py --bgn $bgn_date --end $end_date --factor SMT
python main.factors.py --bgn $bgn_date --end $end_date --factor RWTC
python main.to_tsdb.py --type fac --end $end_date

# feature selection
rm -r /var/data/StackData/futures/team/huxo/feature_selection/*
python main.feature_selection.py --bgn $bgn_date_ml --end $end_date

# model and prediciton
rm -r /var/data/StackData/futures/team/huxo/mclrn/*
rm -r /var/data/StackData/futures/team/huxo/prediction/*
python main.mclrn.manage_models.py
python main.mclrn.py --bgn $bgn_date_ml --end $end_date

# signals
rm -r /var/data/StackData/futures/team/huxo/signals/*
rm -r /var/TSDB/FutHot/d01/e/team/huxo/signals/*
python main.signals.py --type single --bgn $bgn_date_sig --end $end_date
python main.to_tsdb.py --type sig --end $end_date

# single model simulation and evaluation
rm -r /var/data/StackData/futures/team/huxo/simulations/*
rm -r /var/data/StackData/futures/team/huxo/evaluations/*
python main.simulations.py --type single --bgn $bgn_date_sim --end $end_date
python main.evaluations.py --type single --bgn $bgn_date_sim --end $end_date

# portfolios signals, simulation and evaluation
python main.signals.py --type portfolio --bgn $bgn_date_sig --end $end_date
python main.simulations.py --type portfolio --bgn $bgn_date_sim --end $end_date
python main.evaluations.py --type portfolio --bgn $bgn_date_sim --end $end_date
