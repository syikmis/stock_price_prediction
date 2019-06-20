import pickle
from pathlib import Path

import pandas as pd

import data.utils.web_scrappers as ws

DATA_DIR = Path("../data/data")
COM_DATA_DIR = DATA_DIR / "DAX30"
PKL_DIR = DATA_DIR / "PKL_DIR"
DAX_DATA_PKL = PKL_DIR / "DAX30.data.pkl"
DAX_DATA_CSV = DATA_DIR / "DAX30.csv"


def path_to_string(path):
    return "/".join(path.parts)


def compute_dax_df():
    """
    returns main dataframe with stocks data of all dax companies
    :return: dataframe with stocks data
    """

    tickers = ws.get_tickers()
    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        ticker = ticker.rstrip()
        try:
            df = get_com_as_df(ticker)
        except IOError:
            print("No .csv found for {}".format(ticker))
            continue
        df.reset_index(inplace=True)
        df.set_index("Date", inplace=True)
        df.rename(columns={"Adj Close": ticker}, inplace=True)
        df.drop(["Open", "High", "Low", "Close", "Volume"], 1, inplace=True)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.merge(df)

    # print(main_df.tail())
    save_dax_as_csv(main_df)
    save_dax_as_pkl(main_df)


def save_com_as_csv(df, ticker):
    """
    Saves company data as .csv

    :param df: new fetched dataframe from yahoo
    :param ticker: ticker symbol

    """
    path = path_to_string(COM_DATA_DIR) + "/{}.csv".format(ticker)
    df.to_csv(path)
    print("Saved {} data to {}".format(ticker, path))


def save_dax_as_csv(df):
    """
    Saves dax data as .csv
    param df: DataFrame

    """
    path = path_to_string(DATA_DIR) + "/DAX30.csv"
    df.to_csv(path)
    print("Saved DAX30 data to{}".format(path))


def save_dax_as_pkl(df):
    path = path_to_string(DAX_DATA_PKL)
    with open(path, "wb") as f:
        pickle.dump(df, f)
        print("Saved DAX30 data to{}".format(path))


def get_dax_as_pkl():
    path = path_to_string(DAX_DATA_PKL)
    with open(path, "wb") as f:
        pickle.load(f)


def get_dax__as_df():
    if DAX_DATA_PKL.exists():
        path = path_to_string(DAX_DATA_PKL)

        return pd.read_pickle(path)
    else:
        compute_dax_df()


def get_com_as_df(ticker):
    df = pd.read_csv("{}/{}.csv".format(path_to_string(COM_DATA_DIR), ticker))
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    return df


def refresh_dax_data():
    # First get list from Wikipedia with all ticker symbols and name
    ws.get_com_tickers_names()
    # Fetch and save .csv for each com
    ws.get_com_data()
    # compute, save as pickle and return DataFrame
    compute_dax_df()


def refresh_com_data(ticker):
    ws.get_com_data(ticker)
