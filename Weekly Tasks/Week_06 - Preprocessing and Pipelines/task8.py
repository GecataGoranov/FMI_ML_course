import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error


def main():
    music_df = pd.read_csv("../DATA/music_clean.csv", index_col = 0)

    X = music_df.drop(columns = ["energy"])
    y = music_df["energy"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

    linreg_pipeline = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("linreg", LinearRegression())
        ]
    )

    ridge_pipeline = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha = 0.1, random_state = 42))
        ]
    )

    linreg_pipeline.fit(X_train, y_train)
    ridge_pipeline.fit(X_train, y_train)

    linreg_y_pred = linreg_pipeline.predict(X_test)
    ridge_y_pred = ridge_pipeline.predict(X_test)

    print("Linear Regression Test Set RMSE:", root_mean_squared_error(y_test, linreg_y_pred))
    print("Ridge Test Set RMSE:", root_mean_squared_error(y_test, ridge_y_pred))


if __name__ == "__main__":
    main()