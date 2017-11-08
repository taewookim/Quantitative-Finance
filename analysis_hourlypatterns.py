from datetime import datetime, timedelta
from backfill_data import get_data
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc
from matplotlib.finance import volume_overlay
from matplotlib.dates import num2date
from matplotlib.dates import date2num
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import dontbuffer
import argparse

import imageio
from sklearn.preprocessing import MinMaxScaler
# figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
def plot_with_price(quotes, save_to_file=None, min_max=[]):

	# fig = plt.figure()
	
	fig, ax = plt.subplots(figsize=(30, 15))

	# ax.set_xlim([1,2])

	# ax.set_ylim(min_max)

	candlestick2_ohlc(ax,quotes['open'],quotes['high'],quotes['low'],quotes['close'],width=0.6)


	xdate = [i for i in quotes.index]



	ax.xaxis.set_major_locator(ticker.MaxNLocator(6))

	def mydate(x,pos):
	    try:
	        return xdate[int(x)]
	    except IndexError:
	        return ''

	ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))

	ax2 = ax.twinx()
	# ax2.set_ylim([0,5000])
	# Plot the volume overlay
	bc = volume_overlay(ax2, quotes['open'], quotes['close'], quotes['volume'], colorup='g', alpha=0.1, width=1)
	ax2.add_collection(bc)

	fig.autofmt_xdate()
	fig.tight_layout()

	# plt.show()
	if(save_to_file is not None):
		plt.savefig(save_to_file)


def get_saturday_of_week(which_dt):
	idx = (which_dt.weekday() + 1) % 7 # MON = 0, SUN = 6 -> SUN = 0 .. SAT = 6
	sun = which_dt - timedelta(idx)
	return sun

def next_weekday(d, weekday):
	# https://stackoverflow.com/questions/6558535/find-the-date-for-the-first-monday-after-a-given-a-date

    days_ahead = weekday - d.weekday()
    if days_ahead <= 0: # Target day already happened this week
        days_ahead += 7
    return d + timedelta(days_ahead)


	
def main(currency, num_days_to_lookback=30, resample="15Min"):	

	df,_,_ = get_data(currency=currency,
		num_days_to_lookback=num_days_to_lookback,
		resample=resample,
		starting_when = datetime.now()
		)
	df.index = pd.to_datetime(df.index)
	df["volume"] = MinMaxScaler().fit_transform(df["volume"])


	first_date = pd.to_datetime(df.index[0])
	last_date = pd.to_datetime(df.index[-1])
	starting = next_weekday(first_date,0)

	filenames = []

	min_max = [ df["close"].min(), df["close"].max() ]
	while True:

		ending = starting + timedelta(days=1)
		
		df_subset = df.ix[ starting : ending ]
		
		print("[{}] Start {} - End {} - length({})".format(starting.strftime("%a"), starting, ending, len(df_subset)))

		file="time_volatility_analysis/daily_hour/{}-{}-{}-{}.jpg".format(currency, resample, starting.date(), starting.strftime("%a"))


		starting+=timedelta(days=1)
		if(len(df_subset) in [0,1]):
			print("SKipping since df length {}".format(len(df)))
			continue
		if(starting > last_date):
			return

		plot_with_price(df_subset, save_to_file=file, min_max=min_max)
		filenames.append(file)

	images=[]
	for filename in filenames:
		images.append(imageio.imread(filename))

	video="time_volatility_analysis/daily_hour/_ANIMATED_{}-{}.gif".format(currency, resample)
	imageio.mimsave(video, images)


def parse_args():
	parser = argparse.ArgumentParser(
		description='Bid/Ask Line Hierarchy',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	parser.add_argument('--num_days_to_lookback', '-n', action='store',
						required=False, default=30, help='number of days to look back in backtest using Oanda live data')

	parser.add_argument('--resample', '-r', action='store',
						required=False, default="5Min", help='resample period')

	parser.add_argument('--currency', '-c', action='store',
						required=False, default="EUR_USD", help='currency')

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()

	main(args.currency, num_days_to_lookback=args.num_days_to_lookback, resample=args.resample)

	print("*" * 50)
	print("Saved to time_volatility_analysis/daily_hour/")
	print("*" * 50)

