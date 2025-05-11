import numpy as np
import pandas as pd


def main():
    cars_advanced_df = pd.read_csv("../DATA/cars_advanced.csv", index_col = 0)

    cars_ordered_df = cars_advanced_df.sort_values("cars_per_cap", ascending = False)
    
    for lab, row in cars_ordered_df.iterrows():
        print(lab, ': ', row["cars_per_cap"], sep = "")


if __name__ == "__main__":
    main()