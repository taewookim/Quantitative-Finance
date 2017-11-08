from __future__ import (absolute_import, division, print_function, unicode_literals)

from backfill_data import get_data, get_all_currencies
from datetime import datetime, timedelta
#import datetime
from dateutil.relativedelta import relativedelta
#import datetime  # For datetime objects
import os
import operator
import collections
import logging
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import argparse
import backtrader as bt
import dontbuffer
import backtrader.analyzers as btanalyzers
import backtrader.indicators as btind
import pandas as pd
import numpy as np
import time
import collections
import pickle
import logging
#import copy
import configparser
from pprint import pprint, pformat
from get_unitsize import get_size
from tk_utils import crossOver,pandas_tf_to_backtrader_tf, to_dict, dict_recursively, peek_object_variables, banner, MyParser, convert_dict_datatypes, parseString, parseInt, get_unique_id, parseFloat, convert_data_type, write_csv_with_cols, write_dict_to_csv, from_utc_to_local, now_in_open_session
from price_analysis import csv_to_pandas, calc_support_resistance, linear_regression, calc_percentage, get_support_resistance, is_sr_scalpable, kelly_criterion, fx_precision, current_time_volatility_is_safe, percentage_change, get_price_delta_volatility
import sys
from kma import KMA

def parse_args():
	parser = argparse.ArgumentParser(
		description='Bid/Ask Line Hierarchy',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	parser.add_argument('--num_days_to_lookback', '-n', action='store',
						required=True, help='number of days to look back in backtest using Oanda live data')

	parser.add_argument('--config', '-c', action='store',
						required=True, help='INI config file')


	parser.add_argument('--override', '-o', action='store',
						required=False, help="Overide values in config file, kwargs in key=value format")

	parser.add_argument('--quiet', '-q', action='store_true',
						required=False, default=False, help="Dont print anything")

	parser.add_argument('--pairs', '-p', action='store',
						required=True, default=None, help="comma separated currency pairs")


	######################################################################################
	# $ python3 fx_scalp_SR.py  --config=fx_scalp.v1.ini --data smaller.csv --override 'timeframe_0|timeframe="1D",timeframe_0|vol_significance=50000,timeframe_1|timeframe="1W"'
	######################################################################################

	return parser.parse_args()




class MomentumRotation(bt.Strategy):

	params = (
		( 'timeframe_0_tf', '1Min'),
		('is_backtest', "local"),
		('num_days_to_lookback', 30),
		('currency', 'EUR_USD'),
		('size', 2000),
		('leverage', 10),
		('account', '12345'),
		('apikey', '12345')
	)


	kma = {}
	crossover={}
	intersects={}

	def __init__(self):		

		count=0
		for d in self.getdatanames():
			# self.kma[d] = btind.ExponentialMovingAverage(self.getdatabyname(d), period=10)
			self.kma[d] = KMA(self.getdatabyname(d))
			self.crossover[d] = btind.CrossOver(self.kma[d].kma, self.datas[count].close)
			self.intersects[d] = 0
			count+=1

	def notify_trade(self, trade):
		pass

	def notify_order(self, order):
		pass

	def next(self):
		# input("NEXT")
		for d in self.getdatanames():
			# input("{} {}".format(d, self.crossover[d]))
			if(self.crossover[d] == 1.0 or self.crossover[d] == -1.0 ):
				# input("CROSS")
				self.intersects[d] += 1

	def stop(self):

		# print("*" * 50)
		for d in self.getdatanames():
			final_analysis = [
				("strategy"		, os.path.basename(__file__)),
				("timeframe_0_tf", self.p.timeframe_0_tf), 
				("currency", d),
				("total", len(self)),
				("mean_intersects", self.intersects[d])			
			]

			final_analysis = collections.OrderedDict(final_analysis)
			# pprint(final_analysis)
			str_to_format = "{}," * len(final_analysis.keys())            
			print(str_to_format.format(*final_analysis.values()))


def runBacktest(conf_ini):

	# pprint(conf_ini)
	# sys.exit(1)
	config_tf = conf_ini["timeframes"]["timeframe_0_tf"]
	timeframe, compression   = pandas_tf_to_backtrader_tf(config_tf)
	
	# print(config_tf)
	# input("::::")

	logging.info("*	" * 30)
	logging.info("Running backtest - {} {}".format(timeframe, compression))

	cerebro = bt.Cerebro()

	num_days_to_lookback 	= conf_ini["general"]["num_days_to_lookback"]
	resample 				= conf_ini["timeframes"]["timeframe_0_tf"]

	

	pairs = args.pairs.split(",")

	cdata = get_all_currencies(num_days_to_lookback=num_days_to_lookback, resample=resample, pairs=pairs)

	# print(cdata)

	for currency, currency_data in cdata.items():
		csv = currency_data["csv"]
		# print("{} {} {}".format(currency, csv, resample))
		bt_feed = bt.feeds.GenericCSVData(
		    name=currency,
		    dataname=csv,
		    timeframe=timeframe,
		    datetime=0,
		    open=1,
		    high=2,
		    low=3,
		    close=4,
		    volume=5,
		    openinterest=-1
		)	
		# bt_feed.plotinfo.plot = False
		cerebro.adddata(bt_feed)

	# cerebro.addobserver(bt.observers.Broker)
	# cerebro.addobserver(bt.observers.Trades)
	# cerebro.addobserver(bt.observers.BuySell)

	
	logging.info("Adding strategy")
	#pprint(conf_ini)
	cerebro.addstrategy(MomentumRotation,
		timeframe_0_tf = conf_ini["timeframes"]["timeframe_0_tf"],
		currency=conf_ini["general"]["currency"],
		num_days_to_lookback=conf_ini["general"]["num_days_to_lookback"],
		size = conf_ini["broker"]["size"],
		leverage = conf_ini["broker"]["leverage"],
		is_backtest = "local",
		
	)

	logging.info("Running cerebro")
	
	cerebro.run()


	#cerebro.plot(style='bar')
	# cerebro.plot()


if __name__ == '__main__':

	
	
	
	args = parse_args()

	if( not os.path.exists(args.config) ):
		print("Missing config file")
		sys.exit(1)

	config = MyParser()
	config.read(args.config)
	conf_ini = config.as_dict()

	# if("JPY" in conf_ini["general"]["currency"]):
	# 	print("Code will probably need refactoriing, especially in bracket_order / take profit / stop_loss ")
	# 	sys.exit(1)	

	# if( not args.runoanda ):
	# 	#logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s-%(message)s', datefmt="%Y-%m-%d %H:%M:%S")
		
	# 	logging.basicConfig(level=logging.info if (args.quiet) else logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")
		
	# else:
	# 	#logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")
	# 	logging.basicConfig(level=logging.info, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")

	logging.Formatter.converter = time.gmtime
	logging.basicConfig(level=logging.CRITICAL if (args.quiet) else logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")

	if args.override is not None:
		logging.info("Override: {}".format(args.override))
		items = args.override.replace(" ", "").split(",")
		
		for conf_override in items:
			section, variable_kv    = conf_override.split("|")
			variable_k, variable_v  = variable_kv.split("=")            
			conf_ini[ section ][ variable_k ] = variable_v
	
	# logging.info(pformat(conf_ini, indent=4))
	# sys.exit(1)
	# input("Press enter to continue")
	# pprint(conf_ini)

	conf_ini["general"]["num_days_to_lookback"] = int(args.num_days_to_lookback)

	logging.info("RUNNING LOCAL BACKTEST")
	runBacktest(conf_ini)
	