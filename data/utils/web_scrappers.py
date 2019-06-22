import datetime as dt
import os
import pickle
from pathlib import Path

import bs4 as bs
import pandas_datareader.data as web
import requests

import data.utils.df_loader as dl

DATA_DIR = Path("../data/data")
COM_DATA_DIR = DATA_DIR / "DAX30"
PKL_DIR = DATA_DIR / "PKL_DIR"
COM_NAMES_PKL = PKL_DIR / "DAX30.names.pkl"
COM_TICKERS_PKL = PKL_DIR / "DAX30.tickers.pkl"


def path_to_string(path):
    return "/".join(path.parts)


def get_com_tickers_names():
    resp = requests.get(
        "https://de.wikipedia.org/wiki/DAX#Unternehmen_im_DAX")
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"class": "wikitable sortable"})
    tickers = []
    names = []
    for row in table.findAll("tr")[1:]:
        name = row.findAll("td")[0].text
        ticker = row.findAll("td")[1].text
        tickers.append(ticker.rstrip() + ".DE")
        names.append(name.rstrip())

    save_names(names)
    save_tickers(tickers)


def get_com_data(ticker=None):
    """
    Loads stock data from yahoo and saves it as .csv for each company
    """
    path = path_to_string(COM_DATA_DIR)
    if not os.path.exists(path):
        os.makedirs(path)
    if ticker is None:
        tickers = get_tickers()
    else:
        tickers = [ticker]

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        # just in case your connection breaks, we"d like to save our progress!
        try:
            df = web.DataReader(ticker, "yahoo", start, end)
        except IOError:
            print("No data from yahoo found for: {}".format(ticker))
            continue
        dl.save_com_as_csv(df, ticker)


def get_tickers():
    """
    Returns list with all 30 DAX company tickers
    :return: list with ticker symbols
    """
    with open(path_to_string(COM_TICKERS_PKL), "rb") as f:
        tickers = pickle.load(f)

    return tickers


def get_names():
    """
    Returns list with all 30 DAX company names
    :return: list with all DAX company names
    """
    with open(path_to_string(COM_NAMES_PKL), "rb")as f:
        names = pickle.load(f)
    return names


def save_tickers(tickers):
    path = path_to_string(PKL_DIR)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path_to_string(COM_TICKERS_PKL), "wb") as f:
        pickle.dump(tickers, f)
        print("Saved DAX 30 ticker symbols")


def save_names(names):
    path = path_to_string(PKL_DIR)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path_to_string(COM_NAMES_PKL), "wb") as f:
        pickle.dump(names, f)
        print("Saved DAX 30 company names")


def ticker_to_name(ticker):
    names = list(get_names())
    tickers = list(get_tickers())
    index = tickers.index(ticker)
    name = names.pop(index)
    return name
