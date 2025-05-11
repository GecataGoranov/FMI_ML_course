import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col


def main():
    music_missing_vals = pd.read_json("../DATA/music_dirty_missing_vals.txt")

    print("Shape of input dataframe:", music_missing_vals.shape)
    print("Percentage of missing values:")
    print(music_missing_vals.isna().mean().sort_values(ascending = False))

    cols_with_less_than_5_percent = [col for col in music_missing_vals.columns if music_missing_vals[col].isna().mean() < 0.05]

    print("Columns/Variables with missing values less than 5% of the dataset:", cols_with_less_than_5_percent)

    music_missing_vals = music_missing_vals.dropna(subset = cols_with_less_than_5_percent)

    music_missing_vals["genre"] = music_missing_vals["genre"].apply(lambda x: 1 if x == "Rock" else 0)

    print("First five entries in `genre` column:")
    print(music_missing_vals["genre"].head(5))

    print("Shape of preprocessed dataframe:", music_missing_vals.shape)


if __name__ == "__main__":
    main()