import numpy as np
import pandas as pd


def main():
    names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
    dr =  [True, False, False, False, True, True, True]
    cpc = [809, 731, 588, 18, 200, 70, 45]
    country_codes = ["US", "AUS", "JPN", "IN", "RU", "MOR", "EG"]

    df = pd.DataFrame(data = np.array([names, dr, cpc]).transpose(), columns = ["country", "drives_right", "cars_per_cap"], index = country_codes)
    print(df)


if __name__ == "__main__":
    main()