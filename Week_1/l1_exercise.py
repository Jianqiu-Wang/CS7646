__author__ = 'amilkov3'

import pandas as pd
import matplotlib.pyplot as plt

def get_max_close(symbol):
    df = pd.read_csv('../data/{}.csv'.format(symbol))
    #print df.tail(5)
    #print df[10:21]
    df['Adj Close'].plot()
    plt.show()

    return df['Close'].max()
    #return df['Volume'].mean()


def test_run():
    for symbol in ['AAPL', 'IBM']:
        print 'Max close'
        print symbol, get_max_close(symbol)

if __name__ == "__main__":
    test_run()