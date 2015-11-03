__author__ = 'amilkov3'

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

trades_i = 0

def plot_selected(df, columns, start_index, end_index):
    """Plot the desired columns over index values in the given range."""
    df = df.ix[start_index:end_index, columns]
    plot_data(df)

def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def compute_portvals(start_date, end_date, trades_df, start_val):
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
    symbols = []
    for i, row in trades_df.iterrows():
        if row['Symbol'] not in symbols:
            symbols.append(row['Symbol'])

    prices_symbol = get_data(symbols, pd.date_range(start_date, end_date))

    for symbol in symbols:
        prices_symbol[symbol + ' Shares'] = pd.Series(0, index=prices_symbol.index)
    prices_symbol['Port Val'] = pd.Series(start_val, index=prices_symbol.index)
    prices_symbol['Cash'] = pd.Series(start_val, index=prices_symbol.index)

    for i, row in trades_df.iterrows():
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
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':
            df = df.dropna(subset=['SPY'])

    return df

def get_portfolio_value(prices, allocs, start_val=1):
    """Compute daily portfolio value given stock prices, allocations and starting value.

    Parameters
    ----------
        prices: daily prices for each stock in portfolio
        allocs: initial allocations, as fractions that sum to 1
        start_val: total starting value invested in portfolio (default: 1)

    Returns
    -------
        port_val: daily portfolio value
    """
    # TODO: Your code here
    normed = prices/prices.ix[0,:]
    alloced = normed * allocs
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)
    return port_val


def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    """Calculate statistics on given portfolio values.

    Parameters
    ----------
        port_val: daily portfolio value
        daily_rf: daily risk-free rate of return (default: 0%)
        samples_per_year: frequency of sampling (default: 252 trading days)

    Returns
    -------
        cum_ret: cumulative return
        avg_daily_ret: average of daily returns
        std_daily_ret: standard deviation of daily returns
        sharpe_ratio: annualized Sharpe ratio
    """
    # TODO: Your code here
    daily_returns = (port_val / port_val.shift(1)) - 1
    avg_daily_ret = daily_returns.mean()
    cum_ret = (port_val[-1]/port_val[0]) - 1
    std_daily_ret = daily_returns.std()

    k = np.sqrt(samples_per_year)
    sharpe_ratio = k * np.mean(avg_daily_ret - daily_rf)/std_daily_ret
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

def normalize_data(df):
    return df/df.ix[0,:]

def plot_normalized_data(df, title="Normalized prices", xlabel="Date", ylabel="Normalized price"):
    """Normalize given stock prices and plot for comparison.

    Parameters
    ----------
        df: DataFrame containing stock prices to plot (non-normalized)
        title: plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    #TODO: Your code here
    df = df/df.ix[0,:]
    ax = df.plot(title=title, fontsize=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def plot_data(df, title='Stock prices'):

    ax = df.plot(title=title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    #plt.show()

    # Normalize plot
    #df = df/df[0]

def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return pd.rolling_mean(values, window=window)


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    # TODO: Compute and return rolling standard deviation
    return pd.rolling_std(values, window=window)


def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    # TODO: Compute upper_band and lower_band
    upper_band = rm + 2*rstd
    lower_band = rm - 2*rstd
    return upper_band, lower_band

def plot_short_entries_exits(upper_band, rm_IBM, df, trades_df):

    global trades_i

    find_short_entries = np.pad(np.diff(np.array(df['IBM'] > upper_band).astype(int)),
                                (1, 0), 'constant', constant_values=(0, ))
    short_entries = np.where(find_short_entries == -1)
    short_entries_list = [i.tolist() for i in short_entries][0]
    find_short_exits = np.pad(np.diff(np.array(df['IBM'] > rm_IBM).astype(int)),
                              (1, 0), 'constant', constant_values=(0, ))
    short_exits = np.where(find_short_exits == -1)
    short_exits_list = [i.tolist() for i in short_exits][0]

    current_entry = 0
    current_exit = 0

    i = 0
    j = 0

    while i < len(short_entries_list):
        while short_entries_list[i] < current_exit:
            i += 1
        plt.axvline(df.iloc[short_entries_list[i]].name, color='r')
        trades_df.loc[trades_i] = [df.iloc[short_entries_list[i]].name, 'IBM', 'SELL', 100]
        trades_i += 1
        current_entry = short_entries_list[i]
        while j < len(short_exits_list):
            if short_exits_list[j] > current_entry:
                plt.axvline(df.iloc[short_exits_list[j]].name, color='k')
                trades_df.loc[trades_i] = [df.iloc[short_exits_list[j]].name, 'IBM', 'BUY', 100]
                trades_i += 1
                current_exit = short_exits_list[j]
                break
            j += 1
        i += 1
    return trades_df

def plot_long_entries_exits(lower_band, rm_IBM, df, trades_df):

    global trades_i

    find_long_entries = np.pad(np.diff(np.array(df['IBM'] > lower_band).astype(int)),
                                (1, 0), 'constant', constant_values=(0, ))
    long_entries = np.where(find_long_entries == 1)
    long_entries_list = [i.tolist() for i in long_entries][0][1:]
    find_long_exits = np.pad(np.diff(np.array(df['IBM'] > rm_IBM).astype(int)),
                             (1, 0), 'constant', constant_values=(0, ))
    long_exits = np.where(find_long_exits == 1)
    long_exits_list = [i.tolist() for i in long_exits][0][1:]

    current_entry = 0
    current_exit = 0

    i = 0
    j = 0

    while i < len(long_entries_list):
        while long_entries_list[i] < current_exit:
            i += 1
        plt.axvline(df.iloc[long_entries_list[i]].name, color='g')
        trades_df.loc[trades_i] = [df.iloc[long_entries_list[i]].name, 'IBM', 'BUY', 100]
        trades_i += 1
        current_entry = long_entries_list[i]
        while j < len(long_exits_list):
            if long_exits_list[j] > current_entry:
                plt.axvline(df.iloc[long_exits_list[j]].name, color='k')
                trades_df.loc[trades_i] = [df.iloc[long_exits_list[j]].name, 'IBM', 'SELL', 100]
                trades_i += 1
                current_exit = long_exits_list[j]
                break
            j += 1
        i += 1
    return trades_df

def test_run():

    start_date = '2007-12-31'
    end_date = '2009-12-31'

    dates = pd.date_range(start_date, end_date)

    symbols = ['IBM']

    df = get_data(symbols, dates)

    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm_IBM = get_rolling_mean(df['IBM'], window=20)

    # 2. Compute rolling standard deviation
    rstd_IBM = get_rolling_std(df['IBM'], window=20)

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm_IBM, rstd_IBM)

    # Plot raw SPY values, rolling mean and Bollinger Bands
    ax = df['IBM'].plot(title="Bollinger Bands", label='IBM')
    rm_IBM.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='Upper band', ax=ax)
    lower_band.plot(label='Lower band', ax=ax)

    trades_df = pd.DataFrame(columns=['Dates', 'Symbol', 'Order', 'Shares'])

    # Plot short entries and exits
    trades_df = plot_short_entries_exits(upper_band, rm_IBM, df, trades_df)

    # Plot long entries and exits
    trades_df = plot_long_entries_exits(lower_band, rm_IBM, df, trades_df)
    
    # Sort orders by Date and assign Dates column as the index
    trades_df = trades_df.sort('Dates')
    trades_df = trades_df.set_index('Dates')

    # Set cash start val
    start_val = 10000

    # Process orders
    portvals = compute_portvals(start_date, end_date, trades_df, start_val)
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

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower right')
    plt.show()

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")

test_run()
