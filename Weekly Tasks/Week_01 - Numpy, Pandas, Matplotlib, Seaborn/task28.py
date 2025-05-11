import numpy as np
import pandas as pd


def main():
    cars_advanced_df = pd.read_csv("../DATA/cars_advanced.csv", index_col = 0)

    print("Before:\n", cars_advanced_df)

    cars_advanced_df["COUNTRY"] = [row["country"].upper() for _, row in cars_advanced_df.iterrows()]

    print("After:\n", cars_advanced_df)


if __name__ == "__main__":
    main()