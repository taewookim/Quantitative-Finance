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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Bid/Ask Line Hierarchy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    return parser.parse_args()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.CRITICAL, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")

    pairs = ["AUD_JPY", "AUD_USD", "EUR_AUD", "EUR_CHF", "EUR_GBP", "EUR_JPY", "EUR_USD", "GBP_CHF", "GBP_JPY", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY", "XAG_USD",  "XAU_USD"]


    for p in pairs:
        df = pd.read_csv("cron/open_positions/{}/2017-07-07_11-58.csv".format(p))
        df["timestamp"] = pd.to_datetime(df['timestamp'],unit='s')
        df["pct_short"] = 100.0 - df["pct_long"]
        df["net"] = df["pct_long"] - df["pct_short"]
        df["net_change"] = df["net"].pct_change()
        df["price_change"] = df["price"].pct_change()
        df.set_index("timestamp", inplace=True)
        

        df_norm = (df - df.mean()) / (df.max() - df.min())

        print("*" * 50)
        print(p)
        # print(df[["net_change", "price_change"]].corr(method="spearman"))
        df_norm[["net_change", "price_change"]].plot()

        print(df_norm[["net_change", "price_change"]].corr(method="spearman"))

        # print(df.head())
        # print(df.head())
        # plt.show()

