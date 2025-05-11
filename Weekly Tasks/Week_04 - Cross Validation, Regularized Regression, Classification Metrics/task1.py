import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, KFold


def main():
    advertising_and_sales_df = pd.read_csv("../DATA/advertising_and_sales_clean.csv")

    X = advertising_and_sales_df.drop(columns = ["sales", "influencer"])
    y = advertising_and_sales_df["sales"]

    cross_validation_results = cross_validate(LinearRegression(), X, y, cv = KFold(n_splits = 6, shuffle = True, random_state = 5))

    print("Mean:", cross_validation_results["test_score"].mean())
    print("Standard Deviation:", cross_validation_results["test_score"].std())
    print("95% Confidence Interval:", np.percentile(cross_validation_results["test_score"], [2.5, 97.5]))

    plt.plot(cross_validation_results["test_score"])

    plt.title("R^2 per 6-fold split", fontsize = 16)
    plt.xlabel("# Split")
    plt.ylabel("R^2")

    plt.show()

if __name__ == "__main__":
    main()