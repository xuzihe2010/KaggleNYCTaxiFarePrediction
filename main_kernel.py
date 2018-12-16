import numpy as np
import pandas as pd

import os
import time
print(os.listdir("C:/Users/xuzih/Kaggle Data/NYC Taxi Fare/"))

start = time.time(

)
data_raw = pd.read_csv("C:/Users/xuzih/Kaggle Data/NYC Taxi Fare/train.csv", nrows=500_000, parse_dates=["pickup_datetime"])
data_test = pd.read_csv("C:/Users/xuzih/Kaggle Data/NYC Taxi Fare/test.csv", parse_dates=["pickup_datetime"])

print(data_raw.shape)

# Data cleaning and feature engineering functions
def distance_by_long_lat(lon1, lon2, lat1, lat2):
    p = np.pi / 180 # Pi/180
    R = 6371 # Radius of the earth in km
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 2 * R * np.arcsin(np.sqrt(a)) # 2*R*asin...

def minkowski_distance(x_0, x_1, y_0, y_1, p):
    return (abs(x_0 - x_1) ** p + abs(y_0 - y_1) ** p) ** (1/p)

def data_cleaning(df):
    df = df.dropna(how="any",axis="rows")
    return df[(df["dropoff_longitude"] < -72) & (df["dropoff_longitude"] > -75) &
              (df["dropoff_latitude"] > 40) & (df["dropoff_latitude"] < 42) &
              (df["pickup_longitude"] < -72) & (df["pickup_longitude"] > -75) &
              (df["pickup_latitude"] > 40) & (df["pickup_latitude"] < 42) &
              (df["fare_amount"] > 2.5) & (df["fare_amount"] < 100) & (df["passenger_count"] < 7)]

def feature_engineering(df):
    df["abs_longitude"] = np.abs(df.dropoff_longitude - df.pickup_longitude)
    df["abs_latitude"] = np.abs(df.dropoff_latitude - df.pickup_latitude)
    df["year"] = df.pickup_datetime.apply(lambda x : x.year)
    df["month"] = df.pickup_datetime.apply(lambda x : x.month)
    df["day"] = df.pickup_datetime.apply(lambda x : x.day)
    df["weekday"] = df.pickup_datetime.apply(lambda x : x.weekday())
    df["hour"] = df.pickup_datetime.apply(lambda x : x.hour)
    df["distance_manhattan"] = minkowski_distance(df.pickup_longitude, df.dropoff_longitude, df.pickup_latitude, df.dropoff_latitude, 1)
    df["distance_euclidean"] = minkowski_distance(df.pickup_longitude, df.dropoff_longitude, df.pickup_latitude, df.dropoff_latitude, 2)
    df["distance_great_circle"] = distance_by_long_lat(df.pickup_longitude, df.dropoff_longitude, df.pickup_latitude, df.dropoff_latitude)
    return df

def add_fare_bins(df):
    df["fare_bin"] = pd.cut(df.fare_amount, np.linspace(0, 50, 11)).astype(str)
    df.loc[df.fare_bin == "nan", "fare_bin"] = "(50.0+)"
    return df

def add_time_features(df):
    df["pickup_elapsed"] = (df.pickup_datetime - pd.datetime(2009, 1, 1)).dt.total_seconds()
    df["day_of_year"] = df.pickup_datetime.apply(lambda x : x.dayofyear)
    df["days_in_month"] = df.pickup_datetime.apply(lambda x : x.days_in_month)
    df["days_in_year"] = 365 + df.pickup_datetime.apply(lambda x : x.is_leap_year)
    df["frac_day"] = (df.hour + df.pickup_datetime.apply(lambda x : x.minute) / 60 +
                      df.pickup_datetime.apply(lambda x : x.second) / 3600) / 24
    df["frac_week"] = (df.weekday + df.frac_day) / 7
    df["frac_month"] = (df.day + df.frac_day - 1) / df.days_in_month
    df["frac_year"] = (df.day_of_year + df.frac_day - 1) / df.days_in_year
    return df


# Evaluation metrics and function
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

def metrics(train_pred, valid_pred, y_train, y_valid):
    # Root mean squared error
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

    return train_rmse, valid_rmse


def evaluate(model, features, X_train, X_valid, y_train, y_valid):
    # Make predictions
    train_pred = model.predict(X_train[features])
    valid_pred = model.predict(X_valid[features])

    # Get metrics
    train_rmse, valid_rmse = metrics(train_pred, valid_pred, y_train, y_valid)

    print(f'Training:   rmse = {round(train_rmse, 3)}')
    print(f'Validation: rmse = {round(valid_rmse, 3)}')

# Data cleaning and feature engineering
data_raw = data_cleaning(data_raw)
data_raw = feature_engineering(data_raw)
data_raw = add_fare_bins(data_raw)
data_raw = add_time_features(data_raw)

data_test = feature_engineering(data_test)
data_test = add_time_features(data_test)

# Train a random forest model as the benchmark.
time_cols = ["frac_day", "frac_week", "frac_year", "pickup_elapsed"]
feature_cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "distance_great_circle", "abs_longitude",
                "abs_latitude", "passenger_count"] + time_cols

# Define sklearn GradientBoosting regressor and Randomized search cv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

params_grid = {
    'loss': ['ls', 'lad', 'huber', 'quantile'],
    'learning_rate': [0.2 * x for x in range(1, 6)],
    'n_estimators': list(range(100, 1000, 100)) + list(range(1000, 2200, 200)),
    'subsample': [0.1 * x for x in range(1, 10)],
    'min_samples_split': range(2, 10, 2),
    'min_samples_leaf': range(1, 20, 2),
    'max_depth': range(1, 10, 2),
    'max_features': ['auto', 'sqrt', None] + [0.1 * x for x in range(1, 10)]
}

gbr = GradientBoostingRegressor(random_state=1000003, n_iter_no_change=5)
rscv = RandomizedSearchCV(estimator=gbr, param_distributions=params_grid,
                          n_iter=100, n_jobs=-1, scoring="neg_mean_squared_error",
                          verbose=1, random_state=1000003)
end = time.time()
print('start training: ', end - start)

# Run hyperparameter seletion.
start = time.time()
rscv.fit(data_raw[feature_cols], data_raw.fare_amount)
end = time.time()

print("Hyperparameter selection time in minutes: ", (end - start)/60)
print(rscv.best_estimator_)
print(rscv.best_params_)