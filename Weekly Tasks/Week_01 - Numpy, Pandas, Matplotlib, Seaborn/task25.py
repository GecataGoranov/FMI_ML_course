import numpy as np
import pandas as pd


def main():
    cars_advanced_df = pd.read_csv("../DATA/cars_advanced.csv", index_col = 0)

    print(cars_advanced_df[cars_advanced_df["drives_right"] == True])
    print(cars_advanced_df[cars_advanced_df["cars_per_cap"] > 500]["country"])

    print(cars_advanced_df[(cars_advanced_df["cars_per_cap"] >= 10) & (cars_advanced_df["cars_per_cap"] <= 80)]["country"])

    print("Alternative:\n", cars_advanced_df[cars_advanced_df["cars_per_cap"].between(10, 80)]["country"])


if __name__ == "__main__":
    main()