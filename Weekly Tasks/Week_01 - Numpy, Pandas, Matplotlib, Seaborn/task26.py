import numpy as np
import pandas as pd


def main():
    cars_advanced_df = pd.read_csv("../DATA/cars_advanced.csv", index_col = 0)

    for lab, row in cars_advanced_df.iterrows():
        print('Label is "', lab, '"', sep = "")
        print("Row contents:")
        print(row)


if __name__ == "__main__":
    main()