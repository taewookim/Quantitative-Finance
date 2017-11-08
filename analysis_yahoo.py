import os
import logging
import sys  # To find out the script name (in argv[0])
import argparse
import pandas as pd
import numpy as np
import time
import pickle
import logging
from pprint import pprint, pformat
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from os import listdir
from os.path import isfile, join
import pickle
import matplotlib.pyplot as plt
import quandl
import datetime

quandl.ApiConfig.api_key = 'trKLym1kMFxe-3eCFq2G'

if __name__ == '__main__':
    df     = quandl.get("EOD/HD")

    df = df[["Adj_Close"]] # look at price only for now

    factors = [
        #"FRED/BASE", # St. Louis Adjusted Monetary Base
        "FRED/DFF", # Effective Federal Funds Rate
        "FRED/UNRATE", # Civilian Unemployment Rate
        "FRED/CPIAUCSL", # Consumer Price Index for All Urban Consumers: All Items
        # ... so on.. you get the idea
    ]

    for col in factors:
        df[col] = quandl.get(col)

    del(df["Adj_Close"]) # we don't want corrrelation with it self
    df = df.resample("1D") # since some data is monthly / weekly
    df = df.interpolate(method='linear') # Good practice?
    df.dropna(inplace=True)
    
    #scale data for PCA
    from sklearn.preprocessing import StandardScaler,scale
    df[ df.columns ] = StandardScaler().fit_transform(df[df.columns])
    
    pca = PCA()
    pca.fit_transform(df)

    variance_ratios = pca.explained_variance_ratio_
    factors_and_pca = dict(zip(df.columns, variance_ratios ))

    pprint (factors_and_pca)

    plt.bar(range(len(factors_and_pca)), factors_and_pca.values(), align='center')
    plt.xticks(range(len(factors_and_pca)), factors_and_pca.keys(),  rotation='vertical')
    plt.tight_layout()
    plt.show()

