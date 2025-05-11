import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


def main():
    advertising_df = pd.read_csv("advertising_and_sales_clean.csv")

    X = advertising_df.drop(columns = ["sales", "influencer"])
    y = advertising_df["sales"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Predictions: ", predictions[:2])
    print("Actual Values: ", list(y_test[:2]))

    print("R^2: ", model.score(X_test, y_test))
    print("RMSE: ", root_mean_squared_error(y_test, predictions))

    

if __name__ == "__main__":
    main()