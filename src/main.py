from data.utils.df_loader import refresh_com_data, refresh_dax_data
from data.utils.web_scrappers import get_tickers, get_names, ticker_to_name
from ml.learn import do_regression
from plotting.utils import plotter as plt


def main():
    opening = "Hello and welcome to JIYANs \"Stock Analysis\"\n"

    print(opening)

    mes = "There are two possibilites to use this programm:\n " \
          "    - Perform regression on a dax company (1)\n     - Perform analyse over all DAX30 (2)\n"
    print(mes)

    choice = int(input("Please enter 1 or 2 for your choice: "))

    while choice != 1 and choice != 2:
        choice = int(input("Please enter 1 or 2 for your choice: "))

    refresh = str(input("Do you want to refresh the data? [y|N]: "))
    while refresh != "y" and refresh != "N":
        refresh = str(input("Do you want to refresh the data? [y|N]: "))

    if choice == 1:

        tickers = list(get_tickers())
        names = list(get_names())

        fmt = '{:<8}{:<20}{}'
        print("˜\n")
        print(fmt.format('', 'TICKER', 'COMPANY'))
        for i, (ticker, name) in enumerate(zip(tickers, names)):
            print(fmt.format(i + 1, ticker, name))

        com = str(input("Please enter DAX30 company ticker symbol: ")).upper()

        while com not in tickers:
            com = str(input("Please enter correct DAX30 company ticker symbol: ")).upper()

        if com is not None:

            if refresh == "y":
                refresh_com_data(com)

            name = ticker_to_name(com)
            do_regression(ticker=com, name=name, forecast=120)
            plt.plot_100avg(com, name)
            plt.plot_exp_return(com, name)
            plt.plot_ohlc(com, name)

    else:
        if refresh == "y":
            refresh_dax_data()
        plt.plot_dax()


if __name__ == "__main__":
    main()
