import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error


def main():
    data = pd.read_csv("../DATA/bike_sharing.csv")

    data_ohe = pd.get_dummies(data, columns=["season", "weather"], drop_first=True)

    data_ohe["datetime"] = pd.to_datetime(data_ohe["datetime"])
    data_ohe["year"] = data_ohe["datetime"].dt.year
    data_ohe["month"] = data_ohe["datetime"].dt.month
    data_ohe["day"] = data_ohe["datetime"].dt.day
    data_ohe["hour"] = data_ohe["datetime"].dt.hour
    
    X = data_ohe.drop(columns=["count", "datetime"])
    y = data_ohe["count"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    gradient_boosting = GradientBoostingRegressor(
        max_depth=4,
        n_estimators=200,
        subsample=0.9,
        max_features=0.75,
        random_state=2
    )
    gradient_boosting.fit(X_train, y_train)

    y_pred = gradient_boosting.predict(X_test)
    print("Test set RMSE:", np.round(root_mean_squared_error(y_test, y_pred), 2))

    """
    It's a bit different
    """



if __name__ == "__main__":
    main()