import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def main():
    advertising_and_sales_df = pd.read_csv("../DATA/advertising_and_sales_clean.csv")

    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    X_train, X_test, y_train, y_test = train_test_split(advertising_and_sales_df.drop(columns = ["sales", "influencer"]), advertising_and_sales_df["sales"], test_size = 0.3, random_state = 42)

    results = {}

    for alpha in alphas:
        ridge = Ridge(alpha = alpha)
        ridge.fit(X_train, y_train)

        results[alpha] = ridge.score(X_test, y_test)

    print("Ridge scores per alpha:", results)
    # print(type(list(results.keys())))

    plt.plot(list(results.keys()), list(results.values()))

    plt.title("R^2 per alpha", fontsize = 16)
    plt.xlabel("Alpha")
    plt.ylabel("R^2")

    plt.ylim(bottom = 0.99, top = 1)

    plt.show()

    # I don't think we have overfitting, because the model performs well on the test set as well
    # No, we don't have underfitting, judging by the R^2 values
    # In this case, heavy penalization does almost nothing


if __name__ == "__main__":
    main()