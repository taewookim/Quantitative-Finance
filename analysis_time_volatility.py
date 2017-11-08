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
import threading
#import copy
import configparser
from pprint import pprint, pformat
from os import listdir
from os.path import isfile, join
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import scipy.stats as stats
import math
from backfill_data import get_data
import seaborn as sns; sns.set()
from pykalman import KalmanFilter

#sns.palplot(sns.color_palette("RdBu_r", 7))

DIR="time_volatility_analysis"

def parse_args():
    parser = argparse.ArgumentParser(
        description='Bid/Ask Line Hierarchy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--clear', '-c', action='store_true',
                        required=False, default=False, help='clear cache')

    parser.add_argument('--num_days_to_lookback', '-n', action='store',
                        required=False, default=365, help='# of days to correlate')

    parser.add_argument('--resample', '-r', action='store',
                        required=False, default="1H", help='resample to period (default: 1H)')

    return parser.parse_args()


def plot_and_save_heatmap(pivot, title=None, filename=None):
    ax = sns.heatmap(pivot, cmap="RdBu_r")
    fig.suptitle(title, fontsize=20)    
    fig.savefig(filename)
    print("Saved to {}".format(filename))
    plt.clf()

def fill_nonexistent_columns(pivot):
    for i in range(7):
        if i not in pivot:
            pivot[i] = np.nan

    #  return pivot

def get_volatility_stats(currency="EUR_USD", 
    resample="1H", 
    num_days_to_lookback=365, 
    dont_plot_and_return=False):

    logging.info("Fetchin data for {} and resampling to {}".format(currency, resample))
    df,_,_ = get_data(currency=currency,
        num_days_to_lookback=num_days_to_lookback,
        resample=resample
        )
    df.index = pd.to_datetime(df.index)
    # print(df.head())
    # print(df.index)
    df["day"] = df.index.weekday
    df["time"] = df.index.hour
    df["change"] = df["close"].diff().shift(-1)
    df["volatility"] = df["high"] - df["low"]            


    kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance=1,
                      transition_covariance=.01)

    # Use the observed values of the price to get a rolling mean
    df["kf"], _ = kf.filter(df["close"].values)
    # print(df[["close", "kf"]].head(n=30))
    # input(">")

    # chop a few since it takes Kalman filter to catch up
    df = df.iloc[30:]
    # print(df[["close", "kf"]].head(n=30))
    # input(">")

    df["close_before"] = df["close"].shift(1)
    df["close_after"] = df["close"].shift(-1)
    df.dropna(inplace=True)

    intersects = []
    for index, row in df.iterrows():
        intersects.append( 
            1 if (
                (row["close_before"] < row["kf"] <= row["close_after"]) or
                (row["close_before"] > row["kf"] >= row["close_after"])
            ) else 0
        )

    df["kf_intersect"] = intersects
    # print(df[["close", "close_before", "close_after", "kf", "kf_intersect"]].head(n=30))

    # df[["close", "kf", "kf_intersect"]].plot()
    # plt.show()
    # input(">")
    #print((bid["close"] - ask["close"]).head())
    # print((bid["close"] ).head())
    # print((ask["close"]).head())
    # input(">")


    # hl_volatility = pd.pivot_table(df, 
    #     values='volatility', index=['time'],
    #     columns=['day'], aggfunc=np.mean)
    # sns.heatmap(hl_volatility, cmap="RdBu_r")
    # fig.suptitle('{} High Low Volatility'.format(p), fontsize=20)
    # filename = "{}/{}-HL-volatility.jpg".format(DIR, p)    
    # fig.savefig(filename)
    # print("Saved to {}".format(filename))
    # plt.clf()        

    hl_volatility = pd.pivot_table(df.copy(), 
        values="volatility", index=['time'],
        columns=['day'], aggfunc=np.mean)

    fill_nonexistent_columns(hl_volatility)

    # price_change_pivot = pd.pivot_table(df, 
    #     values='change', index=['time'],
    #     columns=['day'], aggfunc=np.mean)
    # ax = sns.heatmap(price_change_pivot, cmap="RdBu_r")
    # fig.suptitle('{} Price Change'.format(p), fontsize=20)
    # filename =     
    # fig.savefig(filename)
    # print("Saved to {}".format(filename))
    # plt.clf()

    price_change = pd.pivot_table(df.copy(), 
        values="change", index=['time'],
        columns=['day'], aggfunc=np.mean)

    fill_nonexistent_columns(price_change)

    mean_reversion = pd.pivot_table(df.copy(), 
        values="kf_intersect", index=['time'],
        columns=['day'], aggfunc=np.sum)

    fill_nonexistent_columns(mean_reversion)

    
    volume = pd.pivot_table(df.copy(), 
        values="volume", index=['time'],
        columns=['day'], aggfunc=np.sum)

    fill_nonexistent_columns(volume)

    if(dont_plot_and_return):
        return hl_volatility, price_change, mean_reversion, volume


    plot_and_save_heatmap(
        hl_volatility, 
        title='{} HL Change'.format(currency), 
        filename="{}/{}_{}_{}_hlchange.jpg".format(DIR, currency, datetime.today().strftime('%Y-%m-%d'), num_days_to_lookback)
    )

    plot_and_save_heatmap(
        price_change,
        title='{} Price Change'.format(currency), 
        filename="{}/{}_{}_{}_pricechange.jpg".format(DIR, currency, datetime.today().strftime('%Y-%m-%d'), num_days_to_lookback)
    )

    plot_and_save_heatmap(
        mean_reversion, 
        title='{} Mean Reversion'.format(currency), 
        filename="{}/{}_{}_{}_meanreversion.jpg".format(DIR, currency, datetime.today().strftime('%Y-%m-%d'), num_days_to_lookback)
    )

    plot_and_save_heatmap(
        volume, 
        title='{} Volume'.format(currency), 
        filename="{}/{}_{}_{}_volume.jpg".format(DIR, currency, datetime.today().strftime('%Y-%m-%d'), num_days_to_lookback)
    )

if __name__ == '__main__':
    # logging.basicConfig(level=logging.CRITICAL, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s', datefmt="%H:%M:%S")
    args = parse_args()

    # pairs = ["EUR_USD", "USD_JPY", "GBP_USD","AUD_USD","USD_CHF","USD_CAD","EUR_JPY","EUR_GBP"]
    # pairs = ["GBP_USD"]
    pairs = ["AUD_CAD", "AUD_CHF", "AUD_HKD", "AUD_JPY", "AUD_NZD", "AUD_SGD", "AUD_USD", "CAD_CHF", "CAD_HKD", "CAD_JPY", "CAD_SGD", "CHF_HKD", "CHF_JPY", "CHF_ZAR", "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_CZK", "EUR_DKK", "EUR_GBP", "EUR_HKD", "EUR_HUF", "EUR_JPY", "EUR_NOK", "EUR_NZD", "EUR_PLN", "EUR_SEK", "EUR_SGD", "EUR_TRY", "EUR_USD", "EUR_ZAR", "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_HKD", "GBP_JPY", "GBP_NZD", "GBP_PLN", "GBP_SGD", "GBP_USD", "GBP_ZAR", "HKD_JPY", "NZD_CAD", "NZD_CHF", "NZD_HKD", "NZD_JPY", "NZD_SGD", "NZD_USD", "SGD_CHF", "SGD_HKD", "SGD_JPY", "TRY_JPY", "USD_CAD", "USD_CHF", "USD_CNH", "USD_CZK", "USD_DKK", "USD_HKD", "USD_HUF", "USD_JPY", "USD_MXN", "USD_NOK", "USD_PLN", "USD_SAR", "USD_SEK", "USD_SGD", "USD_THB", "USD_TRY", "USD_ZAR", "ZAR_JPY"]

    logging.info("Correlating for {} days".format(args.num_days_to_lookback))

    try:
        os.makedirs(DIR)
    except FileExistsError:
        pass 

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    # threads=[]

    # for p in pairs:
    #     t = threading.Thread(
    #         target=get_volatility_stats, 
    #         args =(p, args.resample, args.num_days_to_lookback)
    #         )
    #     t.start()
    #     threads.append(t)

    # for t in threads:
    #     t.join()

    #################
    # thread -safe 
    #################
    [ get_volatility_stats(currency=p, resample=args.resample, num_days_to_lookback=args.num_days_to_lookback) for p in pairs ]

