import os
import logging
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import argparse
import backtrader as bt
import dontbuffer
import backtrader.analyzers as btanalyzers
import pandas as pd
import numpy as np
import time
import scipy.stats as stats
import pickle
import logging
#import copy
import configparser
from pprint import pprint, pformat
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from os import listdir
from os.path import isfile, join
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import scipy.stats as stats
import math
from statsmodels.tsa.stattools import coint
from backfill_data import batch_backfill, separate_bid_ask_midpoint
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.seasonal import seasonal_decompose

def parse_args():
    parser = argparse.ArgumentParser(
        description='Bid/Ask Line Hierarchy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--clear', '-c', action='store_true',
                        required=False, default=False, help='clear cache')

    parser.add_argument('--days', '-d', action='store',
                        required=False, default=365, help='# of days to correlate')

    parser.add_argument('--resample', '-r', action='store',
                        required=False, default="1H", help='resample to period (default: 1H)')

    return parser.parse_args()

def kfilter(df):

    kf = KalmanFilter(transition_matrices = [1],
        observation_matrices = [1],
        initial_state_mean = 0,
        initial_state_covariance = 1,
        observation_covariance=1,
        transition_covariance=.01)

    state_means, _ = kf.filter(df)
    state_means = pd.Series(state_means.flatten(), index=df.index)
    return state_means


def get_correlations(df, which_columns=[], save_csv=None, append=False):

    cols_to_correlate = df[ which_columns ].dropna(how="any")
    spearman_correlation = cols_to_correlate.corr(method="spearman")
    
    if(save_csv is not None):
        #spearman_correlation.to_csv(save_csv)

        with open(save_csv, 'a' if(append) else 'w') as f:
            spearman_correlation.to_csv(f)


    return spearman_correlation


if __name__ == '__main__':
    # logging.basicConfig(level=logging.CRITICAL, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")
    args = parse_args()

    pairs = ["EUR_USD", "USD_JPY", "GBP_USD","AUD_USD","USD_CHF","USD_CAD","EUR_JPY","EUR_GBP"]
    df=None
    since_when = datetime.now() - timedelta(days=int(args.days))

    logging.info("Correlating for {} days".format(args.days))
    

    try:
        os.makedirs("analysis/")
    except FileExistsError:
        pass

    df_fname = "analysis/kfiltered-{}-days-before-{}.pkl".format(args.days, datetime.today().strftime('%Y-%m-%d'))

    if(args.clear):
        logging.info("Clearing {}".format(df_fname))
        try:
            os.remove(df_fname)
        except FileNotFoundError:
            pass

    if (os.path.exists(df_fname) and os.stat(df_fname).st_size > 0 ):
        logging.info("Getting cached data")
        df = pd.read_pickle(df_fname)
    else:

        how_ohlc={
            'open':'first',
            'high':'max',
            'low' :'min',
            'close': 'last',
            'volume': 'sum'
        } 
        for p in pairs:
            logging.info("Fetchin data for {} and resampling to {}".format(p, args.resample))

            #print(batch_backfill(p, since_when = datetime.now() - timedelta(days=1),  is_practice=False))
            
            # print(since_when)
            # sys.exit(1)
            ask,bid,midpoint = batch_backfill(p, since_when = since_when)
                
            # print(ask.head())
            # print(bid.head())
            # print(midpoint.head())
            # input(">")

            ask = ask.resample(args.resample).agg(how_ohlc)
            bid = bid.resample(args.resample).agg(how_ohlc)
            midpoint = midpoint.resample(args.resample).agg(how_ohlc)



            #print((bid["close"] - ask["close"]).head())
            # print((bid["close"] ).head())
            # print((ask["close"]).head())
            # input(">")
            # midpoint["{}-BA_spread".format(p)]  = ask["close"] - bid["close"]
            # midpoint["{}-HL_spread".format(p)]  = midpoint["high"] - midpoint["low"]
            # midpoint["{}-volume".format(p)]     = midpoint["volume"]
            # midpoint["{}-volatility".format(p)] = midpoint["close"].rolling(window=5).std()
            # midpoint["{}-close".format(p)] = midpoint["close"]
            midpoint["{}-kfilter".format(p)] = kfilter(midpoint["close"])   

            # print(midpoint.isnull().values.any())
            midpoint.dropna(inplace=True)
            # print(midpoint["close"].head())
            decomposition = seasonal_decompose(midpoint["close"])

            midpoint["{}-trend".format(p)]      = decomposition.trend
            midpoint["{}-seasonal".format(p)]   = decomposition.seasonal
            midpoint["{}-residual".format(p)]   = decomposition.resid

            # midpoint["{}-close-delta".format(p)] = midpoint["close"].diff()
            # midpoint["{}-close-pct".format(p)] = midpoint["close"].pct_change()

            midpoint.rename(columns={
                "open"  : "{}-open".format(p), 
                "high"  : "{}-high".format(p),
                "low"   : "{}-low".format(p),
                "close" : "{}-close".format(p),
                "volume": "{}-volume".format(p)
                }, inplace=True)

            if (df is None):
                df = midpoint
            else:
                df = df.join(midpoint, how="outer")
                # df.drop_duplicates(keep='last', inplace=True)

            currency_csv = "all_columns-{}.csv".format(p)
            midpoint.to_csv(currency_csv)
            print(currency_csv)


        logging.info("Writing analysis to file {}".format(df_fname))
        df.to_pickle(df_fname)


    df.dropna(inplace=True)
    # print(df.head())
    

    # df[[c for c in df.columns if c.endswith('kfilter')]].plot()
    # plt.show()
    
    coint_heatmap = [[coint( df["{}-residual".format(x)],df["{}-residual".format(y)])[1]  for x in pairs] for y in pairs]
    # pprint(coint_heatmap)

    A = coint_heatmap
    print('\n'.join(['\t'.join(["%0.3f" % item for item in row]) for row in A]))
    # coint_heatmap = [[coint( df["{}-close".format(x)],df["{}-close".format(y)] ) [1]  for x in pairs] for y in pairs]
    # pprint(coint_heatmap)


    # print(df["EUR_USD-close"].head())
    # decomposition = seasonal_decompose(df["EUR_USD-close"])

    # trend = decomposition.trend
    # seasonal = decomposition.seasonal
    # residual = decomposition.resid

    # plt.subplot(411)
    # plt.plot(df["EUR_USD-close"], label='Original')
    # plt.legend(loc='best')
    # plt.subplot(412)
    # plt.plot(trend, label='Trend')
    # plt.legend(loc='best')
    # plt.subplot(413)
    # plt.plot(seasonal,label='Seasonality')
    # plt.legend(loc='best')
    # plt.subplot(414)
    # plt.plot(residual, label='Residuals')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()