import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

def main():
    advertising_df = pd.read_csv("advertising_and_sales_clean.csv")
    print(advertising_df.head(2))

    f, ax = plt.subplots(1, 3)

    ax[0].scatter(advertising_df["tv"], advertising_df["sales"])
    ax[0].set_xlabel("Tv")
    ax[0].set_ylabel("Sales")

    ax[1].scatter(advertising_df["radio"], advertising_df["sales"])
    ax[1].set_xlabel("Radio")
    ax[1].set_ylabel("Sales")

    ax[2].scatter(advertising_df["social_media"], advertising_df["sales"])
    ax[2].set_xlabel("social_media")
    ax[2].set_ylabel("Sales")

    f.tight_layout()

    print("Feature with highest correlation (from visual inspection): TV")

    model = LinearRegression()

    model.fit(advertising_df[["radio"]], advertising_df["sales"])

    predictions = model.predict(advertising_df[["radio"]])

    print(predictions[:5])

    fig, axis = plt.subplots()

    axis.scatter(advertising_df["radio"], advertising_df["sales"])
    axis.plot(advertising_df["radio"], predictions, color = "r")

    plt.show()

    
if __name__ == "__main__":
    main()