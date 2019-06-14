import datetime
from collections import Counter

import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform, randint
from sklearn import svm, neighbors
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from ml.preprocessing import get_reg_data, get_cls_data
from plotting.utils import plotter as plt


def do_classification(ticker):
    X, y, df = get_cls_data(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf = VotingClassifier([("lsvc", svm.LinearSVC()),
                            ("knn", neighbors.KNeighborsClassifier()),
                            ("rfor", RandomForestClassifier())])

    confidence = clf.score(X_test, y_test)
    print("Predictions for : {}".format(ticker))
    print("accuracy:", confidence)
    predictions = clf.predict(X_test)
    print("predicted class counts:", Counter(predictions))
    print()
    print()
    return confidence


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def do_regression(ticker, forecast):
    print("STOCK DATA PREDICTION COM: {}".format(ticker))
    X, y, df, X_data, data = get_reg_data(ticker, forecast)
    regr_1 = make_pipeline(PolynomialFeatures(3), Ridge())
    regr_2 = neighbors.KNeighborsRegressor(n_neighbors=2, weights="distance")
    regr_3 = DecisionTreeRegressor(max_depth=4)
    regr_4 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                               n_estimators=3000)
    regr_5 = XGBRegressor(random_state=42, objective="reg:squarederror")

    selector = RFE(estimator=regr_4, n_features_to_select=5)
    X_new = selector.fit_transform(X, y)
    features = data.columns.values
    print("Features in Dataset: \n{}".format(features))
    features_for_prediction = selector.get_support()
    indicies = [i for i in range(0, len(features_for_prediction)) if
                features_for_prediction[i]]
    columns_for_prediction = [features[i] for i in indicies]
    print("Features used for preditcion based on RFE: \n{}".format(
        columns_for_prediction))
    # select features for prediction
    X_data = X_data[:, indicies]
    X_train, X_test, y_train, y_test = train_test_split(X_new, y,
                                                        test_size=0.3)

    clf = VotingRegressor([("poly3", regr_1),
                           ("knn", regr_2),
                           ("dt_reg", regr_3),
                           ("adb_reg", regr_4),
                           ("xgb_reg", regr_5)])

    param_dist = {"knn__n_neighbors": sp_randint(1, 5),
                  "dt_reg__max_depth": sp_randint(4, 7),
                  "dt_reg__min_samples_split": sp_randint(2, 10),
                  "dt_reg__max_features": sp_randint(2, 5),
                  "adb_reg__base_estimator__max_depth": sp_randint(4, 7),
                  "adb_reg__base_estimator__min_samples_split": sp_randint(2,
                                                                           10),
                  "adb_reg__base_estimator__max_features": sp_randint(2, 5),
                  "adb_reg__n_estimators": sp_randint(1, 5000),
                  "xgb_reg__colsample_bytree": uniform(0.7, 0.3),
                  "xgb_reg__gamma": uniform(0, 0.5),
                  "xgb_reg__learning_rate": uniform(0.03, 0.3),  # default 0.1
                  "xgb_reg__max_depth": randint(2, 6),  # default 3
                  "xgb_reg__n_estimators": randint(100, 150),  # default 100
                  "xgb_reg__subsample": uniform(0.6, 0.4)
                  }
    print(clf.get_params().keys())

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=5, cv=10, n_jobs=-1, verbose=5)
    random_search.fit(X_train, y_train)
    report(random_search.cv_results_)

    # clf.fit(X_train, y_train)
    confidence = random_search.score(X_test, y_test)
    print("Accuracy:", confidence)
    forecast = random_search.predict(X_data)
    data = data[["Adj Close"]]
    data = data.rename(columns={"Adj Close": "EOD"})
    data["Forecast"] = forecast[:]
    print("Predicted class counts:", Counter(forecast))
    print()
    print()

    df["Forecast"] = np.nan
    # df = save_forecast(df, forecast)

    plt.plot_forecast(data, ticker)


def save_forecast(df, forecast):
    last_date = df.iloc[-1].name

    last_unix = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()
    next_unix = last_unix + datetime.timedelta(days=1)
    for i in forecast:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        df.loc[str(next_date)] = [np.nan for _ in
                                  range(len(df.columns) - 1)] + [i]

    return df
