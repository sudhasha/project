import math
from matplotlib import style
import matplotlib.pyplot as plt
import os as os
import json
import pandas as pd
import fix_yahoo_finance as yf  
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

## Data set up
asset_price_plots_dir = "{}/asset_price_plots/".format(os.getcwd())
#os.mkdir("{}/asset_price_plots".format(current_dir)) if not os.path.exists("{}/stock_price_plots".format(current_dir)) else None


#format(ticker_list)".csv"

def read_jason():
    with open("stockdatainfo1.json", 'r') as ticker_list_f:
      ticker_list_str = ticker_list_f.read()
      asset_data_info = json.loads(ticker_list_str)
    return asset_data_info

asset_data_info = read_jason()
ticker = asset_data_info["stock_tickers"]

asset_filename = "assetdata.csv"

"""if os.path.exists(asset_filename):
    print("File already exists.",asset_filename)
    pass
else:"""
df = yf.download(ticker, "2018-01-01", "2019-01-01")
df.to_csv("{}".format(asset_filename))
df = pd.read_csv("{}".format(asset_filename))

df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
df = df.ix[2:]
#df.dropna(inplace=True)

df_date = df.iloc[:, 0].copy(deep=True)
df_date = pd.DataFrame(df_date.values, columns=["Date"])

original_df_dict =  {}
preprocessed_data_dict = {}

for ticker in asset_data_info["stock_tickers"]:
    feature_list = asset_data_info["features_list"]
    i = df[feature_list]
    i.reset_index(drop=True, inplace=True)
    original_df_dict[ticker] = pd.concat([df_date, i], axis=1)
                    
useful_features = ["Adj Close", "HL_PCT", "PCT_change","Return"]
for ticker, original_df in original_df_dict.items():
    #ticker_domain = ticker_symbol.split("/")[0]
    #feature_list = asset_data_info["features_list"][ticker_domain]
    #preprocessed_feature_list = list(map(lambda x, x1: "{} - {}".format(x, x1), [ticker_symbol] * len(feature_list), feature_list))
    preprocessed_df = original_df[feature_list].copy(deep=True)
    df['HL_PCT'] = (pd.to_numeric(df["High"]) - pd.to_numeric(df["Low"])) / pd.to_numeric(df['Adj Close']) * 100.0
    preprocessed_df = df
    df["Return"] = pd.to_numeric(df["Adj Close"]) / pd.to_numeric(df["Adj Close"]).shift(1) - 1
    preprocessed_df = df
    df["PCT_change"] = (pd.to_numeric(df["Close"]) - pd.to_numeric(df["Open"])) / pd.to_numeric(df["Open"]) * 100.0
    preprocessed_df = df
    #preprocessed_feature_list = useful_features
    preprocessed_df = preprocessed_df[useful_features]
    preprocessed_df.dropna(inplace=True)
    future_prediction_pcnt =3
    preprocessed_df["label"] = preprocessed_df["Adj Close"]
    X_forecast = np.array(preprocessed_df.drop(["label"], 1))
    forecast_out = int(math.ceil(0.01 * len(preprocessed_df)))
    preprocessed_df = preprocessed_df.iloc[0: int((1 - future_prediction_pcnt* 0.01) * len(preprocessed_df)), :]
    preprocessed_df["label"] = preprocessed_df["label"].shift(-forecast_out)
    preprocessed_df.dropna(inplace=True)
    X = np.array(preprocessed_df.drop(["label"], 1))
    X = X[:-forecast_out] ##Retunrs, HL_PCT, PCT_CHange, Adj Close
    y = np.array(preprocessed_df["label"])
    y = y[:-forecast_out] ##Adj Close
    preprocessed_data_dict[ticker] = [X, X_forecast, y]   

###Splitting the data
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=25)

#Logistics Regression

lm = LogisticRegression(C = 1e6 )
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
plt.plot(X_test,y_test,color = "red")
plt.plot(X_train, y_pred, color = "blue")

