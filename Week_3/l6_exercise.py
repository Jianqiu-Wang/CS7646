__author__ = 'amilkov3'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from l4_exercise import get_data, plot_data

def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:]/df[:-1].values) - 1
    daily_returns.ix[0, :] = 0 #set daily returns for row 0 to 0
    # Easier
    #daily_returns = (df / df.shift(1)) - 1
    return daily_returns

def hist_plot():
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'XOM']
    df = get_data(symbols, dates)
    daily_returns = compute_daily_returns(df)

    daily_returns['SPY'].hist(bins=20, label='SPY')
    daily_returns['XOM'].hist(bins=20, label='XOM')


    #mean = daily_returns['SPY'].mean()
    #print 'mean = ', mean
    #std = daily_returns['SPY'].std()
    #print 'std = ', std

    #plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    #plt.axvline(std, color='r', linestyle='dashed', linewidth=2)
    #plt.axvline(-std, color='r', linestyle='dashed', linewidth=2)
    plt.legend(loc='upper right')
    plt.show()


    print daily_returns.kurtosis()

def scat_plot():
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'XOM', 'GLD']
    df = get_data(symbols, dates)

    daily_returns = compute_daily_returns(df)

    daily_returns.plot(kind='scatter', x='SPY', y='XOM')
    beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['XOM'], 1)
    plt.plot(daily_returns['SPY'], beta_XOM*daily_returns['SPY'] + alpha_XOM, '-', color='r')
    plt.show()

    daily_returns.plot(kind='scatter', x='SPY', y='GLD')
    beta_GLD, alpha_GLD = np.polyfit(daily_returns['SPY'], daily_returns['GLD'], 1)
    plt.plot(daily_returns['SPY'], beta_GLD*daily_returns['SPY'] + alpha_GLD, '-', color='r')
    plt.show()

    print daily_returns.corr(method='pearson')

scat_plot()