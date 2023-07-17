import yfinance as yf
import numpy as np
import pandas as pd


def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Get stock data from Yahoo Finance.
    :param ticker: Stock ticker.
    :param start: Start date.
    :param end: End date.
    :return: Pandas DataFrame.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    df['Return'] = df['Adj_Close'].pct_change()
    df['Log_Close'] = np.log(df['Adj_Close'])
    df['Log_Return'] = df['Log_Close'].diff()
    return df.dropna()
