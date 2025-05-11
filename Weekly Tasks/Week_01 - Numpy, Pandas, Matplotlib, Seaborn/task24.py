import numpy as np
import pandas as pd


def main():
    cars_df = pd.read_csv("../DATA/cars.csv")

    cars_advanced_df = pd.read_csv("../DATA/cars_advanced.csv", index_col = 0)

    print(cars_advanced_df.loc["JPN"])
    print(cars_advanced_df.loc[["AUS", "EG"]])
    print(cars_advanced_df.loc[["MOR"], ["drives_right"]])
    print(cars_advanced_df.loc[["RU", "MOR"], ["country", "drives_right"]])


if __name__ == "__main__":
    main()