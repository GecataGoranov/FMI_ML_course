import numpy as np
import pandas as pd


def main():
    cars_advanced_df = pd.read_csv("../DATA/cars_advanced.csv", index_col = 0)

    print("Before:\n", cars_advanced_df)

    cars_advanced_df["COUNTRY"] = cars_advanced_df["country"].apply(lambda x : x.upper())

    print("After:\n", cars_advanced_df)


if __name__ == "__main__":
    main()