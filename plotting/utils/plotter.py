from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from mpl_finance import candlestick_ohlc
from pandas.plotting import register_matplotlib_converters

import data.utils.df_loader as dl

register_matplotlib_converters()

style.use("ggplot")

DATA_DIR = Path("data/data")
COM_DATA_DIR = DATA_DIR / "DAX30"

PLOT_DIR = Path("plots")

YEARS = mdates.YearLocator()  # every year
MONTHS = mdates.MonthLocator()  # every month
DAYS = mdates.DayLocator()  # every day
YEARS_FMT = mdates.DateFormatter("%Y")
MONTHS_FMT = mdates.DateFormatter("%m")
DAYS_FMT = mdates.DateFormatter("%d")


def path_to_string(path):
    return "/".join(path.parts)


def set_as_index(df, value="Date"):
    df.reset_index(inplace=True)
    df.set_index(value, inplace=True)
    return df


def plot_100avg(ticker):
    df = dl.get_com_as_df(ticker)
    df.index = pd.to_datetime(df.index)
    df["100ma"] = df["Adj Close"].rolling(window=100, min_periods=0).mean()

    """
        Creates Plot with 2 subfigures:
            - a: > 100 day avg[rolling Window]
                 > Adj Close
            - b: > Volume
    """
    ax1 = plt.subplot2grid((8, 1), (0, 0), colspan=1, rowspan=5)
    ax2 = plt.subplot2grid((8, 1), (5, 0), colspan=1, rowspan=3, sharex=ax1)

    ax1.plot(df.index, df["Adj Close"], label=ticker)
    ax1.plot(df.index, df["100ma"], label="100d mvg avg")
    ax1.set_ylabel("Adj Close")
    ax1.set_title("{}".format(ticker[:-3]))
    ax1.xaxis.set_major_locator(YEARS)
    ax1.xaxis.set_major_formatter(YEARS_FMT)
    ax1.xaxis.set_minor_locator(MONTHS)
    ax1.format_xdata = mdates.DateFormatter("%Y-%m-%d")
    ax1.format_ydata = lambda x: "$%1.2f" % x  # format the price.
    ax1.grid(True)

    ax2.bar(df.index, df["Volume"])
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    ax2.axis("auto")

    path = "{}/{}_100avg.png".format(path_to_string(PLOT_DIR), ticker[:-3])
    plt.savefig(path, dpi=300)
    ax1.legend()
    plt.show()


def plot_exp_return(ticker):
    df = dl.get_com_as_df(ticker)
    df.index = pd.to_datetime(df.index)
    returns = (df["Adj Close"] / df["Adj Close"].shift(1)) - 1
    fig1, ax1 = plt.subplots()
    ax1.plot(df.index, returns, label="return")
    ax1.xaxis.set_major_locator(YEARS)
    ax1.xaxis.set_major_formatter(YEARS_FMT)
    ax1.xaxis.set_minor_locator(MONTHS)

    ax1.set_ylabel("Expected return")
    ax1.set_title("{}".format(ticker[:-3]))
    ax1.legend()

    # format the coords message box
    ax1.format_xdata = mdates.DateFormatter("%Y-%m-%d")
    ax1.format_ydata = lambda x: "$%1.2f" % x  # format the price.
    ax1.grid(True)

    path = "{}/{}_exp_return.png".format(path_to_string(PLOT_DIR), ticker[:-3])
    plt.savefig(path, dpi=300)

    fig1.autofmt_xdate()
    plt.show()


def plot_ohlc(ticker):
    df = dl.get_com_as_df(ticker)
    df = set_as_index(df)
    df.index = pd.to_datetime(df.index)
    df_ohlc = df["Adj Close"].resample("10D").ohlc()
    df_volume = df["Volume"].resample("10D").sum()
    df_ohlc.reset_index(inplace=True)
    df_ohlc["Date"] = df_ohlc["Date"].map(mdates.date2num)

    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()
    ax1.set_title("{}".format(ticker[:-3]))
    candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup="g")
    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
    path = "{}/{}_OHLC.png".format(path_to_string(PLOT_DIR), ticker[:-3])
    plt.savefig(path, dpi=300)
    plt.show()


def plot_dax():
    """ Plots correlation table of all DAX companies"""
    df = dl.get_dax__as_df()
    columns_to_drop = df.columns[:2]
    df = df.drop(columns_to_drop, axis=1)
    df.columns = df.columns.values

    df_corr = df.corr()

    data1 = df_corr.values

    fig1, ax1 = plt.subplots()
    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    ax1.tick_params(axis="both", which="major", pad=10, colors="black")
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    path = "{}/DAX30_cor.png".format(path_to_string(PLOT_DIR))
    plt.savefig(path, dpi=300)
    plt.show()


def plot_forecast(df, ticker):
    fig, ax1 = plt.subplots()
    dates = pd.to_datetime(df.index)
    ax1.plot(dates, df["EOD"], label="Adj Close")
    ax1.plot(dates, df["Forecast"], label="Forecast")
    ax1.legend(loc=4)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.set_title("{}".format(ticker[:-3]))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax1.xaxis.set_minor_locator(DAYS)

    ax1.format_ydata = lambda x: "$%1.2f" % x  # format the price.
    ax1.grid(True)
    fig.autofmt_xdate()
    path = "{}/{}_forecast.png".format(path_to_string(PLOT_DIR), ticker[:-3])
    plt.savefig(path, dpi=300)
    fig.show()
