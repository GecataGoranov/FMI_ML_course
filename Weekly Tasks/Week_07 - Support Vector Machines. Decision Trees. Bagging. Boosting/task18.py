import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.tree import DecisionTreeRegressor


def main():
    data = pd.read_csv("../DATA/auto.csv")
    data = pd.get_dummies(data, drop_first=True)

    X = data.drop(columns=["mpg"])
    y = data["mpg"]

    print(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    tree = DecisionTreeRegressor(
        max_depth=4,
        min_samples_leaf=0.26,
        random_state=1
    )

    cv_score = abs(np.mean(cross_validate(tree, X_train, y_train, cv=10, scoring="neg_root_mean_squared_error")["test_score"]))

    print("10-fold CV RMSE:", np.round(cv_score, 2))
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_train)
    print("Train RMSE:", np.round(root_mean_squared_error(y_train, y_pred), 2))


if __name__ == "__main__":
    main()