from data.utils.df_loader import refresh_dax_data, refresh_com_data
# from plotting.utils import plotter as plt
from stock_price_prediction.ml.learn import do_regression


def main():
    opening = "Hello and welcome to JIYANs \"Stock Analysis\"\n"
    print(opening)

    com = "WDI.DE"
    # refresh_dax_data()
    # refresh_com_data(com)
    # plt.plot_100avg("WDI.DE")
    # plt.plot_exp_return("WDI.DE")
    do_regression(ticker=com, forecast=120)


if __name__ == "__main__":
    main()


# TODO: - more features
# TODO: - XGBoost
# TODO: - NeuralNetwork