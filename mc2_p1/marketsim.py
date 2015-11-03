"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import os

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

def compute_portvals(start_date, end_date, orders_file, start_val):
    """Compute daily portfolio value given a sequence of orders in a CSV file.

    Parameters
    ----------
        start_date: first date to track
        end_date: last date to track
        orders_file: CSV file to read orders from
        start_val: total starting cash available

    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """
    # TODO: Your code here
    df = pd.read_csv(orders_file, index_col='Date',
                     parse_dates=True)
    print df
    symbols = []
    for i, row in df.iterrows():
        if row['Symbol'] not in symbols:
            symbols.append(row['Symbol'])

    prices_symbol = get_data(symbols, pd.date_range(start_date, end_date))

    for symbol in symbols:
        prices_symbol[symbol + ' Shares'] = pd.Series(0, index=prices_symbol.index)
    prices_symbol['Port Val'] = pd.Series(start_val, index=prices_symbol.index)
    prices_symbol['Cash'] = pd.Series(start_val, index=prices_symbol.index)

    for i, row in df.iterrows():
        symbol = row['Symbol']
        if row['Order'] == 'BUY':
            prices_symbol.ix[i:, symbol + ' Shares'] = prices_symbol.ix[i:, symbol + ' Shares'] + row['Shares']
            prices_symbol.ix[i:, 'Cash'] -= prices_symbol.ix[i, symbol] * row['Shares']
        if row['Order'] == 'SELL':
            prices_symbol.ix[i:, symbol + ' Shares'] = prices_symbol.ix[i:, symbol + ' Shares'] - row['Shares']
            prices_symbol.ix[i:, 'Cash'] += prices_symbol.ix[i, symbol] * row['Shares']

    for i, row in prices_symbol.iterrows():
        shares_val = 0
        for symbol in symbols:
            shares_val += prices_symbol.ix[i, symbol + ' Shares'] * row[symbol]
        prices_symbol.ix[i, 'Port Val'] = prices_symbol.ix[i, 'Cash'] + shares_val

    return prices_symbol.ix[:, 'Port Val']

def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2011-01-05'
    end_date = '2011-01-20'
    start_date = '2011-01-14'
    end_date = '2011-12-14'
    orders_file = os.path.join("orders", "orders2.csv")
    start_val = 1000000
    compute_portvals(start_date, end_date, orders_file, start_val)


    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a $SPX-only reference portfolio to get stats
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")


if __name__ == "__main__":
    test_run()
