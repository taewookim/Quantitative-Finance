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
from talib.abstract import *
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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
		required=False, default=3, help='number of days to look back in backtest using Oanda live data')

	parser.add_argument('--resample', '-r', action='store',
		required=False, default="1D", help='resample to period (default: 1H)')

	return parser.parse_args()

def get_indicators(currencies, period, dont_dropna=False):
	currencies_indicators = {}

	# print(currencies)
	# input(">>>>")
	for i in currencies:

		# print(currencies[i]["close"].head(5).mean())

		# print(currencies[i].isnull().sum())
		# c = currencies[i]["close"].as_matrix().tolist()
		# print(c)
		# print(type(c))
		features =  pd.DataFrame(SMA(currencies[i], timeperiod=5))
		# import sys; sys.exit
		features.columns = ['sma_5']
		features['sma_10'] = pd.DataFrame(SMA(currencies[i], timeperiod=10))
		features['mom_10'] = pd.DataFrame(MOM(currencies[i],10))
		features['wma_10'] = pd.DataFrame(WMA(currencies[i],10))
		features['wma_5'] = pd.DataFrame(WMA(currencies[i],5))
		features = pd.concat([features,STOCHF(currencies[i], 
						fastk_period=14, 
						fastd_period=3)],
					axis=1)
		# features['macd'] = pd.DataFrame(MACD(currencies[i], fastperiod=12, slowperiod=26)['macd'])
		features['rsi'] = pd.DataFrame(RSI(currencies[i], timeperiod=14))
		features['willr'] = pd.DataFrame(WILLR(currencies[i], timeperiod=14))
		features['cci'] = pd.DataFrame(CCI(currencies[i], timeperiod=14))
		features['adosc'] = pd.DataFrame(ADOSC(currencies[i], fastperiod=3, slowperiod=10))
		features['pct_change'] = ROC(currencies[i], timeperiod=period)
		features['pct_change'] = features['pct_change'].shift(-period)
		features['pct_change'] = features['pct_change'].apply(lambda x: 1 if (x > 0) else 0 if (x==0) else -1)
		if(dont_dropna is False):
			features.dropna(inplace=True)
		# features = features.iloc[np.where(features.index=='1998-5-5')[0][0]:np.where(features.index=='2015-5-5')[0][0]]

		# print(features["wma_5"])

		currencies_indicators[i] = features
	

	return currencies_indicators

def avg_score(x_train, y_train,x_test,y_test,trees):
    # print("*" * 50)
    # print("x_train")
    # print("*" * 50)
    # print(x_train)

    # print("*" * 50)
    # print("y_train")
    # print("*" * 50)
    # print(y_train)

    # print("*" * 50)
    # print("x_test")
    # print("*" * 50)
    # print(x_test)

    # print("*" * 50)
    # print("y_test")
    # print("*" * 50)
    # print(y_test)
    
    # print("Lengths: {} {} {} {}".format(len(x_train), len(y_train),len(x_test),len(y_test)))

    # input(">")

    if(x_train is None or y_test is None or len(y_test) == 0 or len(x_train) == 0 ):
    	# print("Returning none")
    	return None, None

    accuracy = []
    f1 = []
    rf_model = RandomForestClassifier(trees)
    for i in range(5):
        rf_model.fit(x_train,y_train)
        accuracy.append(rf_model.score(x_test,y_test))
        f1.append(f1_score(y_test,rf_model.predict(x_test), pos_label='1'))
    avg_accuracy = sum(accuracy)/len(accuracy)
    avg_f1 = sum(f1)/len(f1)
    return avg_accuracy, avg_f1

def get_accuracy(currencies, trees, period):
	table_accuracy = pd.DataFrame()
	table_f1 = pd.DataFrame()
	for j in currencies:
		accuracy_values = []
		f1_values = []
		for i in range(1,period+1):
			currencies_indicators = get_indicators(currencies, i)
			train, test = train_test_split(currencies_indicators[j])
			
			accuracy, f1 = avg_score(train.iloc[:,:-1],train.iloc[:,-1],test.iloc[:,:-1],test.iloc[:,-1],trees)

			if(accuracy is None or f1 is None):
				break
			accuracy_values.append(accuracy)
			f1_values.append(f1)
		table_accuracy = pd.concat([table_accuracy, pd.DataFrame({j : accuracy_values})], axis=1)
		table_f1 = pd.concat([table_f1, pd.DataFrame({j : f1_values})], axis=1)
	table_accuracy.index = range(1,period+1)
	table_f1.index = range(1,period+1)
	return table_accuracy, table_f1

currencies = ["EUR_USD", "USD_JPY", "GBP_USD","AUD_USD","USD_CHF","USD_CAD","EUR_JPY","EUR_GBP"]
currencies_indicators = {}

if __name__ == '__main__':
	args = parse_args()

	logging.Formatter.converter = time.gmtime
	logging.basicConfig(level=logging.CRITICAL if (args.quiet) else logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")
	
	
	# close = 
	# print(close)
	#import sys; sys.exit(1)
	currency_closes = {}
	for i in currencies:
		# print("> {}".format(i))
		midpoint, _, _ = get_data(currency=i,
			num_days_to_lookback=args.num_days_to_lookback, 
			resample="1H")
		midpoint["volume"] = midpoint["volume"].apply(lambda x: float(x))
		# print(midpoint.head())
		currency_closes[i] = midpoint

	# currencies_indicators = get_indicators(currency_closes, 1)

	accuracy_table, f1_table = get_accuracy(currency_closes, 300, 30)

	accuracy_table.plot()
	f1_table.plot()
	plt.show()