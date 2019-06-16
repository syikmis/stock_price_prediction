from plotting.utils import plotter as plt
from data.utils.df_loader import refresh_com_data, refresh_dax_data
from ml.learn import do_regression


def main():
    opening = "Hello and welcome to JIYANs \"Stock Analysis\"\n"
    print(opening)

    mes = "There are two possibilites to use this programm:\n " \
          "    - Perform regression on a dax company (1)\n     - Perform analyse over all DAX30 (2)\n"
    print(mes)

    choice = int(input("Please enter 1 or 2 for your Choice: "))

    if choice == 1:
        com = str(input("Please enter DAX30 company ticker symbol: ")).upper()

        if com is not None:

            refresh = int(input("Refresh data [Y: 0, N: 1]: "))
            if refresh == 0:
                refresh_com_data(com)

            do_regression(ticker=com, forecast=120)
            plt.plot_100avg(com)
            plt.plot_exp_return(com)
            plt.plot_ohlc(com)

    else:
        refresh = int(input("Refresh data [Y: 0, N: 1]: "))
        if refresh == 0:
            refresh_dax_data()
        plt.plot_dax()


if __name__ == "__main__":
    main()
