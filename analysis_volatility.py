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
    parser.add_argument('--clear', '-c', action='store_true',
                        required=False, default=False, help='clear cache')

    parser.add_argument('--days', '-d', action='store',
                        required=False, default=365, help='# of days to correlate')

    parser.add_argument('--resample', '-r', action='store',
                        required=False, default="1H", help='resample to period (default: 1H)')

    return parser.parse_args()


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

    df_fname = "analysis/currency-{}-days-before-{}.pkl".format(args.days, datetime.today().strftime('%Y-%m-%d'))

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
            midpoint["{}-BA_spread".format(p)]  = ask["close"] - bid["close"]
            midpoint["{}-HL_spread".format(p)]  = midpoint["high"] - midpoint["low"]
            midpoint["{}-volume".format(p)]     = midpoint["volume"]
            midpoint["{}-volatility".format(p)] = midpoint["close"].rolling(window=5).std()
            midpoint["{}-close".format(p)] = midpoint["close"]
            midpoint["{}-close-delta".format(p)] = midpoint["close"].diff()
            midpoint["{}-close-pct".format(p)] = midpoint["close"].pct_change()

            midpoint.drop(["open", "high", "low", "close", "volume"], axis=1, inplace=True)

            if (df is None):
                df = midpoint
            else:
                df = df.join(midpoint, how="outer")
                df.drop_duplicates(keep='last', inplace=True)

            currency_csv = "all_columns-{}.csv".format(p)
            midpoint.to_csv(currency_csv)
            print(currency_csv)


        logging.info("Writing analysis to file {}".format(df_fname))
        df.to_pickle(df_fname)

    #print(df.columns)
    # df.dropna(inplace=True)
    # print(df.head())
    # print(df.tail())
    # sys.exit(1)    
    #plt.scatter()

    ################################################
    # all columns
    ################################################
    

    get_cpair_corr = True
    get_derivatives = False
    get_close_volatility = False
    plot_bidask_vs_pricedelta = False
    plot_volume_vs_pricedelta = False
    plot_rolling_std = False
    plot_pricedelta_hist = False
    plot_cointegration_heatmap = False
    ################################################
    # Get currency correlation 
    ################################################
    
    if(get_cpair_corr):
        cpair_corr = get_correlations(df, 
            which_columns=[col for col in df.columns if col.endswith('close')], 
            save_csv="cpair_correlation.csv",
            append=False)

        print("cpair_correlation.csv")


    ################################################
    # Get derivative column correlation
    # i.e. correlation between columns with "-" that
    # i specified i.e. BA spread, HL spread, volume, etc.
    ################################################

    if(get_derivatives):    
        append = False
        for p in pairs:
            cols = [ c for c in df.columns if("-" in c and p in c) ]
            derivative_corr = get_correlations(df, 
                which_columns=cols, 
                save_csv="derivative_column_correlation.csv",
                append=append)
            append=True
        
        print("derivative_column_correlation.csv")


    ################################################
    # Get derivative column correlation
    # i.e. correlation between columns with "-" that
    # i specified i.e. BA spread, HL spread, volume, etc.
    ################################################
    

    if(get_close_volatility):
        append = False
        for p in pairs:
            cols = [ c for c in df.columns if(("-" in c and p in c) or c.endswith("volatility")) ]
            volatility = df[ cols ].describe() 
            
            with open("close_volatility.csv", 'a' if(append) else 'w') as f:
                 volatility.to_csv(f)
            append = True

        print("close_volatility.csv")

    
    ###############################
    #
    ###############################

    if(plot_bidask_vs_pricedelta):
        scats = []
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1.0, len(pairs)))

        view_smaller_BA = 0.0010
        # view_smaller_BA = None

        if(view_smaller_BA is not None):
            ba_spread_range = np.arange(0.0, view_smaller_BA,0.0001)
            close_change_range =np.arange(-2.0* view_smaller_BA, 2.0*view_smaller_BA, 0.0001) 
        else:
            ba_spread_range = np.arange(0.0, 0.2000,0.0010)
            close_change_range =np.arange(-1.0, 1.0, 0.1) 

        fig = plt.figure()
        ax = fig.gca()

        ax.set_xticks(ba_spread_range)
        ax.set_yticks(close_change_range)
        ax.set_xticklabels(ba_spread_range, rotation=90)

        for i,p in enumerate(pairs):
            
            # if("EUR_USD" not in p):
            #     continue

            if(view_smaller_BA):
                selected_df = df[df["{}-BA_spread".format(p)] < view_smaller_BA].copy() 
            else:
                selected_df = df.copy()

            X = selected_df["{}-BA_spread".format(p)] 
            Y = selected_df["{}-close-delta".format(p)]
            scats.append(plt.scatter(X, Y, c=colors[i], alpha=0.6) )


        plt.legend(tuple(scats), tuple(pairs), loc="best")
        plt.grid()
        plt.xlabel("bid ask spread")
        plt.ylabel("close change")
        #plt.tight_layout()
        plt.show()

    ###############################
    #
    ###############################

    if(plot_volume_vs_pricedelta):
        scats = []
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1.0, len(pairs)))

        view_smaller_BA = 0.0010
        # view_smaller_BA = None


        volume_range = np.arange(0.0, 1.0, 0.1)
        close_change_range =np.arange(-1.0, 1.0, 0.1) 

        fig = plt.figure()
        ax = fig.gca()

        ax.set_xticks(volume_range)
        ax.set_yticks(close_change_range)
        ax.set_xticklabels(volume_range, rotation=90)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(-1.0, 1.0)

        from sklearn.preprocessing import MinMaxScaler
        for i,p in enumerate(pairs):
            
            # if("EUR_USD" not in p):
            #     continue
            selected_df = df.copy()
            selected_df.dropna(inplace=True)

            X = selected_df["{}-volume".format(p)] 
            X = MinMaxScaler().fit_transform(X)

            Y = selected_df["{}-close-delta".format(p)]
        
            # print(X)
            scats.append(plt.scatter(X, Y, c=colors[i], alpha=0.6) )


        plt.legend(tuple(scats), tuple(pairs), loc="best")
        plt.grid()
        plt.xlabel("volume")
        plt.ylabel("close change")
        #plt.tight_layout()
        plt.show()


    if(plot_cointegration_heatmap):

        coint_heatmap = [[coint( df["{}-close".format(x)],df["{}-close".format(y)])[1]  for x in pairs] for y in pairs]
        pprint(coint_heatmap)

    ############################
    #
    ############################
    if(plot_rolling_std):

        for p in pairs:
            cols = [ c for c in df.columns if(c.endswith("volatility")) ]
            df[cols].plot()
        plt.show()
    # new_df['weekday']   = new_df.index.weekday
    # new_df['hour']      = new_df.index.hour

    #print(df.head(n=1))

    
    ###############################
    #
    ###############################

    if(plot_pricedelta_hist):


        new_df = df.copy()
        new_df.dropna(inplace=True)

        
        append=False
        for p in pairs:

            s = new_df["{}-close-delta".format(p)]
            counts, bins = np.histogram(s,bins=10)
            # print(p)
            # print(pd.Series(counts, index=bins[:-1]))

            #counts_bins = (pd.Series(counts, index=bins[:-1])).to_frame()
            counts_bins = list(zip(bins, counts))
            counts_bins = pd.DataFrame.from_records(counts_bins, columns=["bins", "counts"])
            
            # csv_file= "{}-price_delta_hist.csv".format(p)
            # counts_bins.to_csv(csv_file, index=False)
            # print(csv_file)
            
            with open("price_delta_histogram.csv", 'a' if(append) else 'w') as f:

                f.write(("*" * 30) + "\n" + str(p) + "\n" + ("*" * 30) + "\n")
                counts_bins.to_csv(f)

            append = True

        print("price_delta_histogram.csv")
