import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def main():
    data = pd.read_csv("../DATA/bike_sharing.csv")

    data_ohe = pd.get_dummies(data, columns=["weather"], drop_first=True, dtype=int)

    data_ohe["datetime"] = pd.to_datetime(data_ohe["datetime"])
    data_ohe["year"] = data_ohe["datetime"].dt.year
    data_ohe["month"] = data_ohe["datetime"].dt.month
    data_ohe["day"] = data_ohe["datetime"].dt.day
    data_ohe["hour"] = data_ohe["datetime"].dt.hour
    
    X = data_ohe.drop(columns=["count", "datetime"])
    y = data_ohe["count"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    kf = KFold(n_splits=3, shuffle=True, random_state=2)
    param_grid = {
        "n_estimators" : [100, 350, 500],
        "max_features" : ["log2", "auto", "sqrt"],
        "min_samples_leaf" : [2, 10, 30]
    }
    random_forest = RandomForestRegressor(random_state=2, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=random_forest,
        param_grid=param_grid,
        cv=kf,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error"
    )
    grid_search.fit(X_train, y_train)

    print("Test set RMSE:", -grid_search.score(X_test, y_test))


if __name__ == "__main__":
    main()