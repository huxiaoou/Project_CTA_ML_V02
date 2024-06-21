# CLEAR ALL
rm /var/data/StackData/futures/team/huxo/* -rf
rm /var/TSDB/FutHot/d01/e/team/huxo/* -rf
python config.py

bgn_date="20120104"
bgn_date_ml="20170203"  # machine learning bgn date
bgn_date_sig="20170703" # signal bgn date
bgn_date_sim="20180102" # simulation bgn date
end_date="20240131"     # TSDB Futhot may wrong after this date

# DATA PREPARATION
python main.minbar.py --bgn $bgn_date --end $end_date      # trans futhot m01e from tsdb to ss
python main.alternative.py --bgn $bgn_date --end $end_date # CNY/USD exchange rate; CPI and M2
python main.mbr_pos.py --bgn $bgn_date --end $end_date     # member position, data from WDS
python main.preprocess.py --bgn $bgn_date --end $end_date  # basis, stock, and minor contracts
python main.major.py --bgn $bgn_date --end $end_date       # major contracts and their derivative data
python main.available.py --bgn $bgn_date --end $end_date   # available universe
python main.market.py --bgn $bgn_date --end $end_date      # market and sector return

# TEST RETURN
python main.test_return.py --bgn $bgn_date --end $end_date

# FACTOR CACULATION
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
python main.factors.py --bgn $bgn_date --end $end_date --factor CVP
python main.factors.py --bgn $bgn_date --end $end_date --factor CSP
python main.factors.py --bgn $bgn_date --end $end_date --factor NOI
python main.factors.py --bgn $bgn_date --end $end_date --factor NDOI
python main.factors.py --bgn $bgn_date --end $end_date --factor WNOI
python main.factors.py --bgn $bgn_date --end $end_date --factor WNDOI
python main.factors.py --bgn $bgn_date --end $end_date --factor AMP
python main.factors.py --bgn $bgn_date --end $end_date --factor EXR
python main.factors.py --bgn $bgn_date --end $end_date --factor SMT
python main.factors.py --bgn $bgn_date --end $end_date --factor RWTC

# TO TSDB
python main.to_tsdb.py --type ret --end $end_date
python main.to_tsdb.py --type fac --end $end_date

# feature selection
python main.feature_selection.py --bgn $bgn_date_ml --end $end_date

# model and prediciton
python main.mclrn.manage_models.py
python main.mclrn.py --bgn $bgn_date_ml --end $end_date

# signals
python main.signals.py --type single --bgn $bgn_date_sig --end $end_date
python main.to_tsdb.py --type sig --end $end_date

# single model simulation and evaluation
python main.simulations.py --type single --bgn $bgn_date_sim --end $end_date
python main.evaluations.py --type single --bgn $bgn_date_sim --end $end_date

# portfolios signals, simulation and evaluation
python main.signals.py --type portfolio --bgn $bgn_date_sig --end $end_date
python main.simulations.py --type portfolio --bgn $bgn_date_sim --end $end_date
python main.evaluations.py --type portfolio --bgn $bgn_date_sim --end $end_date
