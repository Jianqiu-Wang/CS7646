__author__ = 'amilkov3'

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

class KNNLearner():
    def __init__(self, k):
        self.k = k

    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

    def query(self, Xtest):
        Y = np.zeros((Xtest.shape[0], 1), dtype='float')

        for i in range(Xtest.shape[0]):
            dist = (self.Xtrain[:, 0] - Xtest[i, 0])**2 + (self.Xtrain[:, 1] - Xtest[i, 1])**2 + (self.Xtrain[:, 2] - Xtest[i, 2])**2
            knn = [self.Ytrain[knni] for knni in np.argsort(dist)[:self.k]]
            Y[i] = np.mean(knn)

        return Y

def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        df = df.dropna(subset=[symbol])

    return df

def training(df, rm, rstd, window, symbol):

    neg_window = window * -1
    price = df[symbol]

    bb_values = (price[window:] - rm)/(2*rstd)

    momentum = df.copy()
    momentum[window:] = (df[window:]/df[:neg_window].values) - 1
    momentum.ix[:window, :] = 0

    daily_returns = df.copy()
    daily_returns[1:] = (df[1:]/df[:-1].values) - 1
    daily_returns.ix[0, :] = 0
    volatility = pd.rolling_std(daily_returns[symbol], window=window)

    X_train = np.zeros((momentum.shape[0], 3))
    X_train[:, 0] = momentum[symbol]
    X_train[:, 1] = bb_values
    X_train[:, 2] = volatility
    X_train[:window, 2] = 0

    Y_train = np.zeros((momentum.shape[0], 1))
    Y_train[:-5, :] = (df[5:]/df[:-5].values) - 1
    Y_train = Y_train[:, 0]

    learner = KNNLearner(k=3)
    learner.addEvidence(X_train, Y_train)
    training_Y = learner.query(X_train)

    predY = [float(x) for x in training_Y]
    rmse = math.sqrt(((Y_train - predY) ** 2).sum()/Y_train.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Y_train)
    print "corr: ", c[0, 1]

    return training_Y, learner

def test(df, rm, rstd, window, symbol, learner):

    neg_window = window * -1
    price = df[symbol]

    bb_values = (price[window:] - rm)/(2*rstd)

    momentum = df.copy()
    momentum[window:] = (df[window:]/df[:neg_window].values) - 1
    momentum.ix[:window, :] = 0

    daily_returns = df.copy()
    daily_returns[1:] = (df[1:]/df[:-1].values) - 1
    daily_returns.ix[0, :] = 0
    volatility = pd.rolling_std(daily_returns[symbol], window=window)

    X_test = np.zeros((momentum.shape[0], 3))
    X_test[:, 0] = momentum[symbol]
    X_test[:, 1] = bb_values
    X_test[:, 2] = volatility
    X_test[:window, 2] = 0

    pred_Y = learner.query(X_test)

    Y_test = np.zeros((momentum.shape[0], 1))
    Y_test[:-5, :] = (df[5:]/df[:-5].values) - 1
    Y_test = Y_test[:, 0]

    predY = [float(x) for x in pred_Y]
    rmse = math.sqrt(((Y_test - predY) ** 2).sum()/Y_test.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Y_test)
    print "corr: ", c[0, 1]

    return pred_Y

def find_entries_exits(df, method):

    #Set long and short shares
    long_shares = 100
    short_shares = 100

    #Set percent over/under price to trigger buy and sell
    percent = 1.03

    long_entries = []
    long_exits = []

    short_entries = []
    short_exits = []

    if method == 'full':

        i = 0
        while i < df.shape[0]:
            if df.ix[i, 'Predicted Y'] > percent * df.ix[i, symbol] or df.ix[i, 'Training Y'] > percent * df.ix[i, symbol]:
                df.ix[i:, symbol + ' Shares'] = df.ix[i:, symbol + ' Shares'] + long_shares
                df.ix[i:, 'Cash'] -= df.ix[i, symbol] * long_shares
                plt.axvline(df.iloc[i].name, color='g')
                long_entries.append(i)
                while i < df.shape[0]:
                    if df.ix[i, 'Predicted Y'] < percent * df.ix[i, symbol] or df.ix[i, 'Training Y'] < percent * df.ix[i, symbol]:
                        df.ix[i:, symbol + ' Shares'] = df.ix[i:, symbol + ' Shares'] - long_shares
                        df.ix[i:, 'Cash'] += df.ix[i, symbol] * long_shares
                        plt.axvline(df.iloc[i].name, color='k')
                        long_exits.append(i)
                        break
                    i += 1
            i += 1

        i = 0

        while i < df.shape[0]:
            if df.ix[i, 'Predicted Y'] < percent * df.ix[i, symbol] or df.ix[i, 'Training Y'] < percent * df.ix[i, symbol]:
                df.ix[i:, symbol + ' Shares'] = df.ix[i:, symbol + ' Shares'] - short_shares
                df.ix[i:, 'Cash'] += df.ix[i, symbol] * long_shares
                plt.axvline(df.iloc[i].name, color='r')
                short_entries.append(i)
                while i < df.shape[0]:
                    if df.ix[i, 'Predicted Y'] > percent * df.ix[i, symbol] or df.ix[i, 'Training Y'] > percent * df.ix[i, symbol]:
                        df.ix[i:, symbol + ' Shares'] = df.ix[i:, symbol + ' Shares'] + short_shares
                        df.ix[i:, 'Cash'] -= df.ix[i, symbol] * short_shares
                        plt.axvline(df.iloc[i].name, color='k')
                        short_exits.append(i)
                        break

                    i += 1
            i += 1

    if method == 'train':

        i = 0

        while i < df.shape[0]:
            if df.ix[i, 'Training Y'] > percent * df.ix[i, symbol]:
                df.ix[i:, symbol + ' Shares'] = df.ix[i:, symbol + ' Shares'] + long_shares
                df.ix[i:, 'Cash'] -= df.ix[i, symbol] * long_shares
                plt.axvline(df.iloc[i].name, color='g')
                long_entries.append(i)
                while i < df.shape[0]:
                    if df.ix[i, 'Training Y'] < percent * df.ix[i, symbol]:
                        df.ix[i:, symbol + ' Shares'] = df.ix[i:, symbol + ' Shares'] - long_shares
                        df.ix[i:, 'Cash'] += df.ix[i, symbol] * long_shares
                        plt.axvline(df.iloc[i].name, color='k')
                        long_exits.append(i)
                        break
                    i += 1
            i += 1

        i = 0

        while i < df.shape[0]:
            if df.ix[i, 'Training Y'] < percent * df.ix[i, symbol]:
                df.ix[i:, symbol + ' Shares'] = df.ix[i:, symbol + ' Shares'] - short_shares
                df.ix[i:, 'Cash'] += df.ix[i, symbol] * long_shares
                plt.axvline(df.iloc[i].name, color='r')
                short_entries.append(i)
                while i < df.shape[0]:
                    if df.ix[i, 'Training Y'] > percent * df.ix[i, symbol]:
                        df.ix[i:, symbol + ' Shares'] = df.ix[i:, symbol + ' Shares'] + short_shares
                        df.ix[i:, 'Cash'] -= df.ix[i, symbol] * short_shares
                        plt.axvline(df.iloc[i].name, color='k')
                        short_exits.append(i)
                        break

                    i += 1
            i += 1

    if method == 'test':

        i = 0

        while i < df.shape[0]:
            if df.ix[i, 'Predicted Y'] > percent * df.ix[i, symbol]:
                df.ix[i:, symbol + ' Shares'] = df.ix[i:, symbol + ' Shares'] + long_shares
                df.ix[i:, 'Cash'] -= df.ix[i, symbol] * long_shares
                plt.axvline(df.iloc[i].name, color='g')
                long_entries.append(i)
                while i < df.shape[0]:
                    if df.ix[i, 'Predicted Y'] < percent * df.ix[i, symbol]:
                        df.ix[i:, symbol + ' Shares'] = df.ix[i:, symbol + ' Shares'] - long_shares
                        df.ix[i:, 'Cash'] += df.ix[i, symbol] * long_shares
                        plt.axvline(df.iloc[i].name, color='k')
                        long_exits.append(i)
                        break
                    i += 1
            i += 1

        i = 0

        while i < df.shape[0]:
            if df.ix[i, 'Predicted Y'] < percent * df.ix[i, symbol]:
                df.ix[i:, symbol + ' Shares'] = df.ix[i:, symbol + ' Shares'] - short_shares
                df.ix[i:, 'Cash'] += df.ix[i, symbol] * long_shares
                plt.axvline(df.iloc[i].name, color='r')
                short_entries.append(i)
                while i < df.shape[0]:
                    if df.ix[i, 'Predicted Y'] > percent * df.ix[i, symbol]:
                        df.ix[i:, symbol + ' Shares'] = df.ix[i:, symbol + ' Shares'] + short_shares
                        df.ix[i:, 'Cash'] -= df.ix[i, symbol] * short_shares
                        plt.axvline(df.iloc[i].name, color='k')
                        short_exits.append(i)
                        break

                    i += 1
            i += 1

    return df

def compute_portvals(full_df):

    for i, row in full_df.iterrows():
        shares_val = 0
        shares_val += full_df.ix[i, symbol + ' Shares'] * row[symbol]
        full_df.ix[i, 'Port Val'] = full_df.ix[i, 'Cash'] + shares_val

    return full_df, full_df.ix[:, 'Port Val']

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

if __name__ == '__main__':

    train_dates = pd.date_range('2008-01-01', '2009-12-31')
    test_dates = pd.date_range('2010-01-01', '2010-12-31')
    price_dates = pd.date_range('2008-01-01', '2010-12-31')

    #Set to either ML4T-399 or IBM
    symbol = 'IBM'

    symbols = [symbol]

    train_df = get_data(symbols, train_dates)
    test_df = get_data(symbols, test_dates)
    full_df = get_data(symbols, price_dates)

    # Set target_df to one of the above dfs
    target_df = test_df
    # Set method to test, train, or full
    method = 'test'
    # Set whether you'd like to include entries and exits
    include_entries_exits = True

    #Set momentum and bollinger window
    window = 20

    train_rm = pd.rolling_mean(train_df[symbol], window=window)
    train_rstd = pd.rolling_std(train_df[symbol], window=window)
    test_rm = pd.rolling_mean(test_df[symbol], window=window)
    test_rstd = pd.rolling_std(test_df[symbol], window=window)

    if method == 'full' or method == 'test':
        training_Y, learner = training(train_df, train_rm, train_rstd, window, symbol)
        pred_Y = test(test_df, test_rm, test_rstd, window, symbol, learner)
    if method == 'train':
        training_Y, learner = training(train_df, train_rm, train_rstd, window, symbol)

    start_val = 10000

    if method == 'train':
        target_df['Training Y'] = 0
        target_df['Training Y'] = training_Y[:, 0]
        target_df['Training Y'] = target_df[symbol] * (target_df['Training Y']+1)

    if method == 'test':
        target_df['Predicted Y'] = 0
        target_df['Predicted Y'] = pred_Y[:, 0]
        target_df['Predicted Y'] = target_df[symbol] * (target_df['Predicted Y']+1)


    if method == 'full':
        target_df['Training Y'] = 0
        target_df['Training Y'][:training_Y.shape[0]] = training_Y[:, 0]
        target_df['Training Y'] = target_df[symbol] * (target_df['Training Y']+1)
        target_df['Training Y'][training_Y.shape[0]:] = np.nan
        target_df['Predicted Y'] = 0
        target_df['Predicted Y'][training_Y.shape[0]:] = pred_Y[:, 0]
        target_df['Predicted Y'] = target_df[symbol] * (target_df['Predicted Y']+1)
        target_df['Predicted Y'][:training_Y.shape[0]] = np.nan

    target_df['Port Val'] = pd.Series(start_val, index=target_df.index)
    target_df[symbol + ' Shares'] = 0
    target_df['Cash'] = pd.Series(start_val, index=target_df.index)

    if include_entries_exits:

        start_date = '2008-01-01'
        end_date = '2010-12-31'

        target_df = find_entries_exits(target_df, method)
        target_df, portvals = compute_portvals(target_df)

        cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

        prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
        prices_SPX = prices_SPX[['$SPX']]  # remove SPY
        portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
        cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

        # Compare portfolio against $SPX
        print
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

    if method == 'full':
        ax = target_df['Training Y'].plot(title=symbol + ' Training Y/Price/Predicted Y', label='Training Y')
        target_df[symbol].plot(label='Price', ax=ax)
        target_df['Predicted Y'].plot(label='Predicted Y', ax=ax)
        ax.legend(loc='upper left')
        plt.show()

    if method == 'train':
        ax = target_df['Training Y'].plot(title=symbol + ' In Sample Entries/Exits', label='Training Y')
        target_df[symbol].plot(label='Price', ax=ax)
        ax.legend(loc='upper left')
        plt.show()

    if method == 'test':
        ax = target_df['Predicted Y'].plot(title=symbol + ' Out Of Sample Entries/Exits', label='Predicted Y')
        target_df[symbol].plot(label='Price', ax=ax)
        ax.legend(loc='upper left')
        plt.show()

    if include_entries_exits:
        df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
        plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")








