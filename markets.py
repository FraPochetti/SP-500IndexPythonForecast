# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 00:29:18 2014

@author: francesco
"""

import pystocks
import datetime
import sys
import pandas as pd
import matplotlib.pyplot as plt
import locale
locale.setlocale(locale.LC_ALL,'en_US.UTF-8')
from pylab import *

fout = 'sp500'
method = 'RF'
best_model = 'sp500_57.pickle'
############ SET PARAMETERS ##################################################
path_datasets = 'path to datasets'
cut = datetime.datetime(1993,1,1)
start_test = datetime.datetime(2014,4,1)
parameters = []
##############################################################################
# IN CASE OF FEATURE AND MODEL SELECTION
maxlags = 10
maxdeltas = 10
folds = 10
#grid = {'C': [0.01, 0.1], 'gamma': [0, 1]}
################################################################################
# AFTER BEST MODEL SELECTION
bestlags = 9
bestdelta = 9
savemodel = False
##############################################################################

if __name__ == "__main__":
    
    ### PIPELINE
    ##########################################################################    
    ## 1- PERFORM FEATURE SELECTION APPLYING RANDOM FOREST TO THE DATA SET.
    ##    THE FUNCTION CAN LOAD DATA FROM THE WEB OR FROM CSV FILES PREVIOUSLY SAVED TO DISK.
    ##    THE OUTPUT IS GOING TO BE A LOG FILE WITH THE RESULT OF CROSS VALIDATION ON TRAIN SET
    
    sys.stdout = open('path to log txt file', 'w')  
    pystocks.performFeatureSelection(maxdeltas, maxlags, fout, cut, start_test, path_datasets, savemodel, method, folds, parameters)             
    
    ##########################################################################
    # 2- CHECK BEST PARAMETERS
    print pystocks.checkModel('path to log txt file')    
    
    ##########################################################################
    # 3- AFTER HAVING SELECTED THE TWO PARAMETERS THAT MAXIMIZE THE ACCURACY ON CV
    #    (MAXDELTA, MAXLAGS) RUN THE MODEL ON THE WHOLE TRAIN SET AND GET A RESULT
    #    ON TEST SET. IF WILLINGLY SAVE THE MODEL TO CPICKLE.
    
    print pystocks.performSingleShotClassification(bestdelta, bestlags, fout, cut, start_test, path_datasets, savemodel, method, parameters)

    ##########################################################################
    # 4- RUN THE TRADING ALGORITHM ON TOP OF THE PREDICTION AND GET RETURNS OF THE BACKTEST
    
    #end_period is the last day of backtesting. The data per prediction where collected in the ]
    #frame 1/1/1990 - 31/08/2014, after that the first 3 years were cut so the final dataframe
    #collects data in the period 1/1/1993 - 31/08/2014. But, as 30-31/08/2014 were Saturday and
    #Sunday the data ventually is restricted to 1/1/1993 - 29/08/2014. As we are predicting what 
    #happens tomorrow (the percentage variation of today closing price of the stock respect to yesterday)
    #so in the end we'll have an end_period whose last day is a day before the actual last trading day.
    end_period = datetime.datetime(2014,8,28) 

    ###### SP500
    symbol = 'S&P-500'
    name = path_datasets + '/sp500.csv'
    prediction = pystocks.getPredictionFromBestModel(9, 9, 'sp500', cut, start_test, path_datasets, 'sp500_57.pickle')[0]
    
    
    bars = pd.read_csv(name, index_col=0, parse_dates=True)    
    bars = bars[start_test:end_period]
 
    signals = pd.DataFrame(index=bars.index)
    signals['signal'] = 0.0
    signals['signal'] = prediction
    signals.signal[signals.signal == 0] = -1
    signals['positions'] = signals['signal'].diff()     

    # Create the portfolio based on the forecaster
    amount_of_shares = 500
    portfolio = pystocks.MarketIntradayPortfolio(symbol, bars, signals, initial_capital = 100000.0, shares = amount_of_shares)
    returns = portfolio.backtest_portfolio()

    # Plot results
    f, ax = plt.subplots(2, sharex=True)
    f.patch.set_facecolor('white')
    ylabel = symbol + ' Close Price in $'
    bars['Close_Out'].plot(ax=ax[0], color='r', lw=3.)    
    ax[0].set_ylabel(ylabel, fontsize=18)
    ax[0].set_xlabel('', fontsize=18)
    ax[0].legend(('Close Price S&P-500',), loc='upper left', prop={"size":18})
    ax[0].set_title('S&P 500 Close Price VS Portofolio Performance (1 April 2014 - 28 August 2014)', fontsize=20, fontweight="bold")
    
    returns['total'].plot(ax=ax[1], color='b', lw=3.)  
    ax[1].set_ylabel('Portfolio value in $', fontsize=18)
    ax[1].set_xlabel('Date', fontsize=18)
    ax[1].legend(('Portofolio Performance. Capital Invested: 100k $. Shares Traded per day: 500+500',), loc='upper left', prop={"size":18})            
    plt.tick_params(axis='both', which='major', labelsize=14)
    loc = ax[1].xaxis.get_major_locator()
    loc.maxticks[DAILY] = 24

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    plt.show()





