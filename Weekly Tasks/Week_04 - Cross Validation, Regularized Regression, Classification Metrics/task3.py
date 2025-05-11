import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso


def main():
    advertising_and_sales_df = pd.read_csv("../DATA/advertising_and_sales_clean.csv")

    X = advertising_and_sales_df.drop(columns = ["sales", "influencer"])
    y = advertising_and_sales_df["sales"]

    lasso = Lasso()
    lasso.fit(X, y)

    coef_dict = dict(zip(X.columns, np.round(lasso.coef_, 4)))

    print("Lasso coefficients per feature:", coef_dict)

    plt.bar(X.columns, lasso.coef_)

    plt.title("Feature importance", fontsize = 16)
    plt.xlabel("Feature")
    plt.ylabel("Importance")

    plt.show()

    # It becomes very apparent, that "tv" is the most important metric, when it comes to predicting "sales"


if __name__ == "__main__":
    main()