bgn_date_ml="20170203"  # machine learning bgn date
bgn_date_sig="20170703" # signal bgn date
bgn_date_sim="20180102" # simulation bgn date
end_date="20240131"     # TSDB Futhot may wrong after this date

# feature selection
python main.feature_selection.py --bgn $bgn_date_ml --end $end_date

# model and prediciton
rm -r /var/data/StackData/futures/team/huxo/mclrn/*
rm -r /var/data/StackData/futures/team/huxo/prediction/*
python main.mclrn.manage_models.py
python main.mclrn.py --bgn $bgn_date_ml --end $end_date

# signals
rm -r /var/data/StackData/futures/team/huxo/signals/*
python main.signals.py --type single --bgn $bgn_date_sig --end $end_date

# translate signals to tsdb
rm -r /var/TSDB/FutHot/d01/e/team/huxo/signals/*
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
