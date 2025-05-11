import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def main():
    data = pd.read_csv("../DATA/auto.csv")

    X = pd.get_dummies(data.drop(columns=["mpg"]), drop_first=True)
    y = data["mpg"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    tree = DecisionTreeRegressor(
        max_depth=8, 
        min_samples_leaf=0.13,
        random_state=3
    )
    tree.fit(X_train, y_train)

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    y_pred_tree = tree.predict(X_test)
    y_pred_linreg = linreg.predict(X_test)

    print("Regression Tree test set RMSE:", np.round(root_mean_squared_error(y_test, y_pred_tree), 2))
    print("Linear Regression test set RMSE:", np.round(root_mean_squared_error(y_test, y_pred_linreg), 2))


if __name__ == "__main__":
    main()