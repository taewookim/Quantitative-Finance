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

def collect_df(mypath ="quandl/"):

    main_df = None
    for f in listdir(mypath):
        filestring =join(mypath, f) 
        if(isfile(filestring) and filestring.endswith(".pkl")):
            df_read = pickle.load( open( filestring, "rb" ) )

            df_read = df_read.ix['2001-01-01':]

            column = filestring .replace(".pkl", "").replace("/", "-")

            df_read.to_csv("tmp-{}.csv".format(column))
            # print("*" * 50)
            # print("{}".format(filestring))
            # print("*" * 50)
            # print(df_read.head())

            df_read = df_read.resample("1D").interpolate(method='linear')

            if(main_df is None):
                main_df = df_read
                main_df.columns = [column]
            else:
                main_df[ column ] = df_read

    return main_df

if __name__ == '__main__':
    df = collect_df()
    #df.to_csv("gigantic.csv")

    # cols = [0, 1,2,4,5,6,7,8,9]
    # df.drop(df.columns[cols],axis=1,inplace=True)
    #df.to_csv("normal.csv")
    df.dropna(inplace=True)

    # print(df.columns)
    # print("*" * 50)
    # print(df.head())
    from sklearn.preprocessing import StandardScaler,scale
    df[ df.columns ] = StandardScaler().fit_transform(df[df.columns])
    
    # print("*" * 50)
    # print(df.head())
    df.to_csv("scaled.csv")
    # sys.exit(1)

    pca = PCA()
    pca.fit_transform(df)

    variance_ratios = pca.explained_variance_ratio_
    factors_and_pca = dict(zip(df.columns, variance_ratios ))

    # effective_components=np.cumsum(np.round(variance_ratios, decimals=4)*100)
    # factors_and_effective_components = dict(zip(df.columns, effective_components ))

    pprint(factors_and_pca) 

    
    plt.bar(range(len(factors_and_pca)), factors_and_pca.values(), align='center')
    plt.xticks(range(len(factors_and_pca)), factors_and_pca.keys(),  rotation='vertical')
    plt.tight_layout()
    plt.show()

    #print(pca.explained_variance_ratio_)

