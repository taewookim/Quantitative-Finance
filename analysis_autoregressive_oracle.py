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
from pandas.io.json import json_normalize
from pandas.tools.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

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

def plot_AR(args, debug=False):
	print("AR Analysis")
	midpoint = get_data(currency=args.currency,
		num_days_to_lookback=args.num_days_to_lookback, 
		resample="1Min")

	'''
	1. print correlation matrix
	'''
	# corr_df = pd.concat([midpoint["close"].shift(1), midpoint["close"]], axis=1)
	# corr_df.columns = ['t-1', 't+1']
	# print(corr_df.corr())

	'''
	2. plot ACF
	'''
	# plot_acf(midpoint["close"], lags=1000)

	'''
	3. plot lag
	'''
	# pd.plotting.lag_plot(midpoint["close"])
	

	''' 
	4. prediction
	'''

	num_predictions=100
	X = midpoint["close"].values

	train, test = X[1:len(X)-num_predictions], X[len(X)-num_predictions:]
	# train autoregression

	startTime = datetime.now()
	
	model 		= AR(train)
	model_fit 	= model.fit()

	print("Train & Fit time: {}".format(datetime.now() - startTime))

	window 		= model_fit.k_ar
	coef 		= model_fit.params

	print('Lag: %s' % model_fit.k_ar)
	print('Coefficients: %s' % model_fit.params)

	# walk forward over time steps in test

	history 	= train[len(train)-window:]
	
	history 	= [history[i] for i in range(len(history))]

	predictions = list()

	for t in range(len(test)):
		length 	= len(history)
		lag 	= [history[i] for i in range(length-window,length)]
		yhat 	= coef[0]
		for d in range(window):
			yhat += coef[d+1] * lag[window-d-1]
		obs 	= test[t]
		predictions.append(yhat)
		history.append(obs)
		# print('predicted=%f, expected=%f' % (yhat, obs))
	
	print(predictions)
	error = mean_squared_error(test, predictions)
	print('Test MSE: {}'.format(error))
	# plot
	plt.plot(test)
	plt.plot(predictions, color='red')
	plt.show()

	# predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
	# for i in range(len(predictions)):
	# 	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
	# error = mean_squared_error(test, predictions)
	# print('Test MSE: %.3f' % error)
	# # plot results
	# plt.plot(test)
	# plt.plot(predictions, color='red')
	# plt.show()

	


if __name__ == '__main__':
	args = parse_args()

	logging.Formatter.converter = time.gmtime
	logging.basicConfig(level=logging.CRITICAL if (args.quiet) else logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")

	plot_AR(args)