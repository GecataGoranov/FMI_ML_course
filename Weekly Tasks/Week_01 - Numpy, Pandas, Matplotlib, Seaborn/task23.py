import numpy as np
import pandas as pd


def main():
    cars_df = pd.read_csv("../DATA/cars.csv")

    print(cars_df)

    cars_df_fixed_index = pd.read_csv("../DATA/cars.csv", index_col = 0)

    print("After setting first column as index:\n", cars_df_fixed_index)




if __name__ == "__main__":
    main()