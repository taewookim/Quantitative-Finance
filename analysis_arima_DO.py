import oandapy
from pprint import pprint
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import dontbuffer
import time
import os
import logging
import os.path  # To manage paths
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
from backfill_data import get_data 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools

def parse_args():
	parser = argparse.ArgumentParser(
		description='Bid/Ask Line Hierarchy',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	parser.add_argument('--quiet', '-q', action='store_true',
						required=False, default=False, help="Dont print anything")

	parser.add_argument('--method', '-m', action='store',
						required=False, default="oanda", help="which COT data to use - oanda or cot")

	parser.add_argument('--num_days_to_lookback', '-n', action='store',
		required=False, default=3650, help='number of days to look back in backtest using Oanda live data')

	parser.add_argument('--currency', '-c', action='store',
		required=False, default="EUR_USD", help='Currency')

	parser.add_argument('--resample', '-r', action='store',
		required=False, default="1D", help='resample to period (default: 1H)')

	return parser.parse_args()

def analysis(y):
	print("ARIMA Analysis")
	
	num_predictions=100

	# Define the p, d and q parameters to take any value between 0 and 2
	p = d = q = range(0, 2)

	# Generate all different combinations of p, q and q triplets
	pdq = list(itertools.product(p, d, q))

	# Generate all different combinations of seasonal p, q and q triplets
	seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

	results=[]
	for param in pdq:
	    for param_seasonal in seasonal_pdq:
	        try:
	            mod = sm.tsa.statespace.SARIMAX(y,
	                                            order=param,
	                                            seasonal_order=param_seasonal,
	                                            enforce_stationarity=False,
	                                            enforce_invertibility=False)

	            results = mod.fit()

	            results.append('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
	        except:
	            continue
	print("*" * 50)
	print("*" * 50)
	print(results)
	

def predict(y):

	mod = sm.tsa.statespace.SARIMAX(y,
		order=(1, 1, 1),
		seasonal_order=(1, 1, 1, 12),
		enforce_stationarity=False,
		enforce_invertibility=False)

	results = mod.fit()

	print(results.summary().tables[1])

	results.plot_diagnostics(figsize=(15, 12))
	plt.show()

	pred = results.get_prediction(start=pd.to_datetime('2017-04-01'), dynamic=False)
	pred_ci = pred.conf_int()
	
	y_forecasted = pred.predicted_mean
	y_truth = y['2017-04-01':]

	# Compute the mean square error
	mse = ((y_forecasted - y_truth) ** 2).mean()
	print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


if __name__ == '__main__':
	args = parse_args()

	logging.Formatter.converter = time.gmtime
	logging.basicConfig(level=logging.CRITICAL if (args.quiet) else logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")

	midpoint, _, _ = get_data(currency=args.currency,
		num_days_to_lookback=args.num_days_to_lookback, 
		resample="1Min")
	print(midpoint.head())
	analysis(midpoint["close"])