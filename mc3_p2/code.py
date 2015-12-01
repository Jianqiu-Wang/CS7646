__author__ = 'amilkov3'

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    learner = KNNLearner(k=3)
    learner.addEvidence(X_train, Y_train)
    training_Y = learner.query(X_train)

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

    return pred_Y

if __name__ == '__main__':

    train_dates = pd.date_range('2008-01-01', '2009-12-31')
    test_dates = pd.date_range('2010-01-01', '2010-12-31')
    price_dates = pd.date_range('2008-01-01', '2010-12-31')

    symbol = 'IBM'

    symbols = [symbol]
    # 'ML4T-399'

    train_df = get_data(symbols, train_dates)
    test_df = get_data(symbols, test_dates)
    full_df = get_data(symbols, price_dates)

    window = 20

    train_rm = pd.rolling_mean(train_df[symbol], window=window)
    train_rstd = pd.rolling_std(train_df[symbol], window=window)
    test_rm = pd.rolling_mean(test_df[symbol], window=window)
    test_rstd = pd.rolling_std(test_df[symbol], window=window)

    training_Y, learner = training(train_df, train_rm, train_rstd, window, symbol)
    pred_Y = test(test_df, test_rm, test_rstd, window, symbol, learner)

    full_df['Training Y'] = 0
    full_df['Predicted Y'] = 0

    full_df['Training Y'][:training_Y.shape[0]] = training_Y[:, 0]
    full_df['Predicted Y'][training_Y.shape[0]:] = pred_Y[:, 0]

    full_df['Training Y'] = full_df[symbol] * (full_df['Training Y']+1)
    full_df['Training Y'][training_Y.shape[0]:] = np.nan
    full_df['Predicted Y'] = full_df[symbol] * (full_df['Predicted Y']+1)
    full_df['Predicted Y'][:training_Y.shape[0]] = np.nan

    ax = full_df['Training Y'].plot(title=symbol + ' Training Y/Price/Predicted Y', label='Training Y')
    full_df[symbol].plot(label='Price', ax=ax)
    full_df['Predicted Y'].plot(label='Predicted Y', ax=ax)
    ax.legend(loc='upper left')
    plt.show()







