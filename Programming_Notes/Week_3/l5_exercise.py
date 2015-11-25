__author__ = 'amilkov3'

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def symbol_to_path(symbol, base_dir="../data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        # TODO: Read and join data for each symbol
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', 'Adj Close'],
                              na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close' : symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':
            df = df.dropna(subset=['SPY'])

    return df

def plot(df):
    ax = df.plot(title='Incomplete Data', fontsize=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.show()

def test_run():

    symbollist = ['JAVA','FAKE1','FAKE2']

    start_date = '2005-12-31'
    end_date = '2014-12-07'

    idx = pd.date_range(start_date, end_date)

    df = get_data(symbollist, idx)
    df.fillna(method='ffill', inplace='TRUE')
    df.fillna(method='bfill', inplace='TRUE')
    plot(df)


test_run()
