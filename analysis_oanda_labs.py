import oandapy
from pprint import pprint
import sys
import argparse
import logging
import pandas as pd
import dontbuffer
import time
import os
import logging
import os.path  # To manage paths
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
from backfill_data import batch_backfill, separate_bid_ask_midpoint
from pandas.io.json import json_normalize
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import coint
apik="3ad86c7125baf271ca0b1be4070f7c79-90291329808b4211f2f43b963ac0fa72"

def parse_args():
	parser = argparse.ArgumentParser(
		description='Bid/Ask Line Hierarchy',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	parser.add_argument('--quiet', '-q', action='store_true',
						required=False, default=False, help="Dont print anything")

	parser.add_argument('--num_days_to_lookback', '-n', action='store',
		required=False, default=3650, help='number of days to look back in backtest using Oanda live data')

	parser.add_argument('--currency', '-c', action='store',
		required=False, default="EUR_USD", help='Currency')

	parser.add_argument('--resample', '-r', action='store',
		required=False, default="1D", help='resample to period (default: 1H)')

	return parser.parse_args()

def get_data():
	args = parse_args()

	logging.Formatter.converter = time.gmtime
	logging.basicConfig(level=logging.CRITICAL if (args.quiet) else logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")

	cache_datafeed = "cache/oandalabs-{}-{}_{}days_{}.pkl".format(args.currency, time.strftime("%Y-%m-%d"), args.num_days_to_lookback, args.resample)

	how_ohlc={
		'open':'first',
		'high':'max',
		'low' :'min',
		'close': 'last',
		'volume': 'sum',
		} 

	since_when = datetime.now() - timedelta(days=int(args.num_days_to_lookback))
	if (os.path.exists(cache_datafeed) is False or os.stat(cache_datafeed).st_size == 0 ):
		logging.info("Caching data feed - {}".format(cache_datafeed))
		_,_,midpoint = batch_backfill(
			args.currency, 
			since_when=since_when
		)
		midpoint = midpoint.resample(args.resample).agg(how_ohlc).dropna()
		midpoint.to_pickle(cache_datafeed)
	else:
		logging.info("Reading cached pickle {}".format(cache_datafeed))
		midpoint = pd.read_pickle(cache_datafeed)
	return midpoint

def main():
	logging.basicConfig(level=logging.INFO, format=' %(asctime)s -%(levelname)s - %(message)s')
	midpoint = get_data()

	oanda = oandapy.API(environment="practice", access_token=apik)
	cot = oanda.get_commitments_of_traders(instrument="EUR_USD")
	cot_df = pd.DataFrame(cot["EUR_USD"]).set_index(["date"])

	####################################################
	# pd.to_datetime(cot_df.index,  unit='s')  => converts int to datetime
	# .normalize() => gets rid of HH:MM:SS
	####################################################
	cot_df.index = pd.to_datetime(cot_df.index,  unit='s').normalize()
	del(cot_df["unit"])

	# print(cot_df.index)
	# print(cot_df.head())
	# sys.exit(1)
	merged = pd.concat([midpoint, cot_df], axis=1, join_axes=[midpoint.index])
	# merged = pd.concat([cot_df], axis=1, join_axes=[midpoint.index])
	merged.dropna(inplace=True)
	

	# print(merged.head(n=5))
	# print(merged.tail(n=5))
	# sys.exit(1)
	merged[["ncl", "oi", "price", "ncs"]] = MinMaxScaler().fit_transform(merged[["ncl", "oi", "price", "ncs"]])
	merged.rename(columns={'ncl': 'noncommercial_long', 'oi': 'open_interest', 'price':"price", "ncs":"noncommercial_short"}, inplace=True)
	
	# print(cot_df.head(n=5))
	# print(cot_df.tail(n=5))

	merged["net"] = merged["noncommercial_long"] - merged["noncommercial_short"]

	# spearman_correlation = merged[["price", 'net']].corr(method="spearman")
	# print(spearman_correlation)
	# with open("cot_correlation.csv", 'w') as f:
	# 	spearman_correlation.to_csv(f)

	# merged[["noncommercial_long", "noncommercial_short", "open_interest", "price", 'net']].plot()
	# merged[["price", 'net', 'open_interest']].plot()
	# print(coint(merged["price"], merged['net']))

	# plt.show()

	'''
	oi: Overall Interest
	ncl: Non-Commercial Long
	price: Exchange Rate
	date: Time returned as a unix timestamp
	ncs: Non-Commercial Short
	unit: Gives the sizes of a single contract in Overall Interest, Non-Commercial Long, and Non-Commercial Short
	'''
	# orderbook = oanda.get_orderbook(instrument="EUR_USD", period="31536000 ")
	# pprint(orderbook)

	'''
	os: Percentage short orders
	ol: Percentage long orders
	ps: Percentage short positions
	pl: Percentage long positions
	rate entry: This is the rate at that specific time
	'''
	# calendar = oanda.get_eco_calendar(instrument="EUR_USD", period="31536000 ")
	# pprint(calendar)

	'''
	title: The title of the event.
	timestamp: Time of the event, returned as a unix timestamp.
	unit: This field describes the data found in the forecast, previous, actual and market fields. Some possible values are: % meaning the data is a percentage, k meaning thousands of units, or blank if there is no data associated with the event.
	currency: This is the currency that is affected by the news event.
	forecast: The forecasted value.
	previous: Shows the value of the previous release of the same event.
	actual: The actual value, this is only available after the event has taken place.
	market: The market expectation of what the value was.
	'''

if __name__ == '__main__':
	main()