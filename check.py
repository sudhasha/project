import math
from matplotlib import style
import matplotlib.pyplot as plt
import os as os
import json
import pandas as pd
import quandl
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor



## Data set up

current_dir = os.getcwd()
asset_price_plots_dir = "{}/asset_price_plots/".format(os.getcwd())
#os.mkdir("{}/asset_price_plots".format(current_dir)) if not os.path.exists("{}/stock_price_plots".format(current_dir)) else None

asset_filename = "assetdata.csv"

def read_jason():
    with open("stockdatainfo.json", 'r') as ticker_list_f:
      ticker_list_str = ticker_list_f.read()
      asset_data_info = json.loads(ticker_list_str)
    return asset_data_info

asset_data_info = read_jason()
for ticker_domain, ticker_list in asset_data_info["stock_tickers"].items():
      asset_tickers = sorted(ticker_list, key=str.lower)
      asset_tickers = list(map(lambda asset_ticker: "{}/{}".format(ticker_domain, asset_ticker), asset_tickers))
      
quandl.ApiConfig.api_key = asset_data_info["KEY"]

if asset_data_info["update_csv"] or not os.path.exists("{}/{}".format(os.getcwd(), asset_filename)):
    df = quandl.get(asset_tickers, start_date="2017-01-01", end_date="2018-12-01", ) 
    df.to_csv("{}".format(asset_filename))
    df = pd.read_csv("{}".format(asset_filename))


## Preprocessing Data
original_df_dict = {}
future_prediction_pcnt = 1
preprocessed_data_dict = {} 
#future_prediction_pcnt = kwargs["future_prediction_pcnt"]

# Copy the date column to another dataframe
df_date = df.iloc[:, 0].copy(deep=True)
df_date = pd.DataFrame(df_date.values, columns=["Date"])

		# Drop date from original dataframe for easier separation stock data based on ticker symbols
for ticker_symbol in asset_tickers:
    feature_list = asset_data_info["features_list"][ticker_domain]
    ticker_symbol_columns = list(map(lambda x, y: "{} - {}".format(x, y), [ticker_symbol] * len(feature_list), feature_list))
    i = df[ticker_symbol_columns]
    original_df_dict[ticker_symbol] = pd.concat([df_date,i], axis = 1)
    
useful_features = ["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]
for ticker_symbol, original_df in original_df_dict.items():
    #ticker_domain = ticker_symbol.split("/")[0]
    #feature_list = asset_data_info["features_list"][ticker_domain]
    preprocessed_feature_list = list(map(
            lambda x, x1: "{} - {}".format(x, x1), [ticker_symbol] * len(feature_list), 
            feature_list))
    preprocessed_df = original_df[preprocessed_feature_list].copy(deep=True)
    if ticker_domain in ["WIKI"]:
        df["{} - HL_PCT".format(ticker_symbol)] = (df["{} - Adj. High".format(ticker_symbol)] - df["{} - Adj. Low".format(ticker_symbol)]) / df["{} - Adj. Low".format(ticker_symbol)] * 100.0
        preprocessed_df = df
        df["{} - PCT_change".format(ticker_symbol)] = (df["{} - Adj. Close".format(ticker_symbol)] - df["{} - Adj. Open".format(ticker_symbol)]) / df["{} - Adj. Open".format(ticker_symbol)] * 100.0
        preprocessed_df = df
        preprocessed_feature_list = list(map(lambda x, x1: "{} - {}".format(x, x1), [ticker_symbol] * len(useful_features), useful_features))
        preprocessed_df = preprocessed_df[preprocessed_feature_list]
    forecast_col_labels = {"WIKI":"{} - Adj. Close".format(ticker_symbol)}
    preprocessed_df.dropna(inplace=True)
    preprocessed_df["label"] = preprocessed_df[forecast_col_labels[ticker_domain]]
    X_forecast = np.array(preprocessed_df.drop(["label"], 1))
    forecast_out = int(math.ceil(future_prediction_pcnt * 0.01 * len(preprocessed_df)))
    preprocessed_df = preprocessed_df.iloc[0: int((1 - future_prediction_pcnt * 0.01) * len(preprocessed_df)), :]
    preprocessed_df["label"] = preprocessed_df["label"].shift(-forecast_out)
    preprocessed_df.dropna(inplace=True)
    X = np.array(preprocessed_df.drop(["label"], 1))
    X = X[:-forecast_out]
    y = np.array(preprocessed_df["label"])
    y = y[:-forecast_out]
    preprocessed_data_dict[ticker_symbol] = [X, X_forecast, y]

##Build Models
built_models_dict = {}
model_scores_dict = {}
saved_models_dir = "saved_models"
models_dict = {
			"Decision Tree Regressor": DecisionTreeRegressor(),
			"Linear Regression": LinearRegression(),
            "Logistic Regression" : LogisticRegression(),
			"Random Forest Regressor": RandomForestRegressor(),
			"SVR": SVR()
            }
parameters_dict = {
			"Decision Tree Regressor": {"max_depth": [200]},
			"Linear Regression": {"n_jobs": [None, -1]},
            "Logistic Regresssion": {"random_state" : [0]},
            "Random Forest Regressor": {"max_depth": [200], 
                               "n_estimators": [100]},
			"SVR": {"kernel": ["rbf", "linear"], "degree": [3], "gamma": ["scale"]}
            }
saved_models_path = "{}/{}".format(os.getcwd(), saved_models_dir)
learning_curve_dir_path = "{}/learning_curve_plots".format(os.getcwd())
os.mkdir(saved_models_path) if not os.path.exists(saved_models_path) else None
os.mkdir(learning_curve_dir_path) if not os.path.exists(learning_curve_dir_path) else None

models_list = ["Logistic Regression"] #["Linear Regression", "Logistic Regression", "Decision Tree Regressor", "Random Forest Regressor", "SVR"]

def get_train_and_test_data(X, y, tscv):
    split_data = []
    for train_indices, test_indices in tscv.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        split_data.append((X_train, X_test, y_train, y_test))
		# Get cross validation score for the last index as it will have the most training data which is good for time
		# series data
        best_split_index = -1
        X_train, X_test, y_train, y_test = split_data[best_split_index]
        print("Last train_data size = {}".format(len(X_train) * 100 / len(X)))
        return X_train, X_test, y_train, y_test

def build_model(model_name, preprocessed_data_dict, force_build):
    model_dict = {}
    model_scores_dict = {}
    curr_dir = os.getcwd()
    for ticker_symbol, preprocessed_data in preprocessed_data_dict.items():
        model_name = "Logistic Regression"
        [X, X_forecast, y] = preprocessed_data
        tscv = TimeSeriesSplit(n_splits=5)
        ticker_symbol = ticker_symbol.replace("/", "_")
        if(model_name == "Logistic Regression"):
                p = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1']}
                labelencoder = preprocessing.LabelEncoder()
                #X = labelencoder.fit(X)
                #X_forecast = labelencoder.fit_transform(X_forecast)
                y = labelencoder.fit_transform(y)
                optimized_model = GridSearchCV(estimator=models_dict[model_name],param_grid = p, cv=tscv)
                #model = make_pipeline(StandardScaler(), optimized_model)
                #X_train, X_test, y_train, y_test = get_train_and_test_data(X, y, tscv)
                #model = model.fit(X_train, y_train)
        else:
            optimized_model = GridSearchCV(estimator=models_dict[model_name], param_grid=parameters_dict[model_name], cv=tscv)
        model = make_pipeline(StandardScaler(), optimized_model)
        X_train, X_test, y_train, y_test = get_train_and_test_data(X, y, tscv)
        model.fit(X_train, y_train)
        pickle_out = open("{}/{}_{}_{}.pickle".format(saved_models_dir, model_name, ticker_symbol, "model"), "wb")
        pickle.dump(model, pickle_out)
        pickle_out.close()
        pickle_in = open("{}/{}_{}_{}.pickle".format(saved_models_dir, model_name, ticker_symbol, "model"), "rb")
        model = pickle.load(pickle_in)
            #model = load_from_pickle_file(model_name, ticker_symbol, "model")
        X_train, X_test, y_train, y_test = get_train_and_test_data(X, y, tscv)
			# Training score
        confidence_score = model.score(X_test, y_test)
			# Plot learning curves
        title = "{}_{}_Learning Curves".format(model_name, ticker_symbol)
        save_file_path = "{}/learning_curve_plots/{}_{}.png".format(curr_dir, model_name, ticker_symbol)
			# Create the CV iterator
        ylim=None
        cv=2
        train_sizes=np.linspace(0.1, 1)
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
										 color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
										 color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.savefig("{}".format(save_file_path))
        plt.close()
            #plot_learning_curve(model, title, X, y, save_file_path, cv=tscv)
			# Cross validation
        cv_scores = cross_validate(model, X=X, y=y, cv=tscv)
        print("Training score for {} = {}".format(ticker_symbol, confidence_score))
        print("Cross validation scores for {} = {}".format(ticker_symbol, cv_scores["test_score"]))
        print("Cross validation score for {} = {} +/- {}".format(ticker_symbol, cv_scores["test_score"].mean(), cv_scores["test_score"].std() * 2))
        print("Cross validation scoring time = {}s".format(cv_scores["score_time"].sum()))
        model_dict[ticker_symbol] = model
        model_scores_dict[ticker_symbol] = confidence_score
    return model_dict, model_scores_dict


for model_name in models_list:
    model_dict, model_scores_dict = build_model(model_name,preprocessed_data_dict , force_build = False)
    built_models_dict[model_name] = model_dict
    model_scores_dict[model_name] = model_scores_dict


"""    forecast_df_dict = forecast_prices.make_predictions(models_dict, preprocessed_data_dict, original_df_dict)
    self.plot_forecast(forecast_df_dict, original_df_dict, future_prediction_pcnt)


if __name__ == "__main__":
  stock_price_prediction = StockPricePrediction()
  stock_price_prediction.main()"""



