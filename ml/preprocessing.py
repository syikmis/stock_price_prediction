from collections import Counter

import numpy as np
import sklearn
from pandas import DataFrame
from sklearn.impute import SimpleImputer

import data.utils.df_loader as dl
import data.utils.web_scrappers as ws


def process_data_for_labels(ticker):
    """
    Computes new columns needed for label generation for specific ticker.
    Getting future values by shifting a column up. Gives future values i days in
    advance.
    :param ticker: Company symbol
    :return: Dataframe with new columns
    """
    hm_days = 7
    df = dl.get_dax__as_df()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        df["{}_{}d".format(ticker, i)] = (df[ticker].shift(-i) - df[
            ticker]) / df[ticker]

    df.fillna(0, inplace=True)

    return df


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def get_cls_data(ticker):
    """
    Firstly computes labels based on new generated columns
    :param ticker: Company symbol
    :return: Features, labels, DataFrame
    """
    tickers = ws.get_tickers()

    df = process_data_for_labels(ticker)

    df["{}_target".format(ticker)] = list(map(buy_sell_hold,
                                              df["{}_1d".format(ticker)],
                                              df["{}_2d".format(ticker)],
                                              df["{}_3d".format(ticker)],
                                              df["{}_4d".format(ticker)],
                                              df["{}_5d".format(ticker)],
                                              df["{}_6d".format(ticker)],
                                              df["{}_7d".format(ticker)]))

    vals = df["{}_target".format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print("Dataspread:", Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df["{}_target".format(ticker)].values
    return X, y, df


def add_new_features(df_org, forecast_out):
    df = df_org.loc[:, ["Adj Close", "Volume"]]
    df["high_low_pct"] = (df_org["High"] - df_org["Low"]) / df_org[
        "Close"] * 100.0
    lbd = lambda x: np.log(x) - np.log(x.shift(1))
    df["change"] = np.log(df_org["Adj Close"]) - np.log(df_org["Adj "
                                                               "Close"].shift(
        1))
    df["pct_change"] = (df_org["Close"] - df_org["Open"]) / df_org[
        "Open"] * 100.0
    df["daily_return"] = (df_org["Close"] / df_org["Open"]) - 1
    df.fillna(method="pad")
    df["Volume"] = np.log(df_org["Volume"])
    # log of 5 day moving average of volume
    df["5d_mean_log"] = df_org["Volume"].rolling(5).mean().apply(
        np.log)
    # daily volume vs. 200 day moving average
    df["volume_mov_avg"] = (df_org["Volume"] / df_org["Volume"].rolling(
        200).mean()) - 1
    # daily closing price vs. 50 day exponential moving avg
    df["close_vs_moving"] = (df_org["Close"] / df_org["Close"].ewm(
        span=50).mean()) - 1
    # z-score
    df["z_score"] = (df_org["Close"] - df_org["Close"].rolling(window=200,
                                                               min_periods=20).mean()) / \
                    df_org["Close"].rolling(window=200, min_periods=20).std()
    df["signing"] = df["pct_change"].apply(np.sign)
    df["plus_minus"] = df["signing"].rolling(20).sum()
    df["label"] = df["Adj Close"].shift(-forecast_out).round(3)
    df["label"] = df["label"].interpolate(limit=3, limit_direction="both")

    df = df.replace([np.inf, -np.inf], np.nan)
    # df = missing_values_transformer(df)
    df.fillna(df.mean(), inplace=True)

    return df


def missing_values_transformer(df):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    np_array = imp_mean.fit_transform(df)
    df = DataFrame.from_records(np_array)
    return df


def get_reg_data(ticker, forecast):
    df = dl.get_com_as_df(ticker)
    forecast_out = int(forecast)  # predict int days into future
    df = add_new_features(df, forecast_out)
    print("Description of data set: \n {}".format(df.describe()))
    X = np.array(df.drop(["label"], 1))
    scaler = sklearn.preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    X_data = X[-forecast_out:]
    X = X[:-forecast_out]

    data = df[-forecast_out:]
    df = df[:-forecast_out]
    y = np.array(df["label"])

    # # Value distrib
    # vals = df["label"].values.tolist()
    # str_vals = [str(i) for i in vals]
    # print("Data spread:", Counter(str_vals))

    return X, y, df, X_data, data
