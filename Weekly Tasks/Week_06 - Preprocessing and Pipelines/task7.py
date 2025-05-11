import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyparsing import line
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso


def main():
    music_df = pd.read_csv("../DATA/music_clean.csv", index_col = 0)

    X = music_df.drop(columns = ["energy"])
    y = music_df["energy"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

    kf = KFold(n_splits = 6, shuffle = True, random_state = 42)

    linear_regression = LinearRegression()
    lasso = Lasso(alpha = 0.1, random_state = 42)
    ridge = Ridge(alpha = 0.1, random_state = 42)

    linear_regression_results = cross_validate(linear_regression, X_train, y_train, cv = kf)
    lasso_results = cross_validate(lasso, X_train, y_train, cv = kf)
    ridge_results = cross_validate(ridge, X_train, y_train, cv = kf)

    plt.boxplot([linear_regression_results["test_score"], ridge_results["test_score"], lasso_results["test_score"]],
                tick_labels = ["Linear Regression", "Ridge", "Lasso"])
    plt.show()

    """
    Judging by the plot, Ridge and Linear Regression perform practically the same
    """


if __name__ == "__main__":
    main()