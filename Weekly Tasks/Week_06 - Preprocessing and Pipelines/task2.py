import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyparsing import col
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score


def main():
    music_df_raw = pd.read_json("../DATA/music_dirty.txt")
    dummies = pd.get_dummies(music_df_raw["genre"], drop_first = True)

    music_df = pd.concat([music_df_raw.drop(columns = ["genre"]), dummies], axis = 1)

    X = music_df.drop(columns = ["popularity"])
    y = music_df["popularity"]

    kf = KFold(n_splits = 6, shuffle = True, random_state = 42)

    ridge = Ridge(alpha = 0.2, random_state = 42)

    score = cross_val_score(ridge, X, y, cv = kf, scoring = "neg_root_mean_squared_error")

    print("Average RMSE:", abs(np.mean(score)))
    print("Standard Deviation of the target array:", y.std())

    """
    The average RMSE is less than the standard deviation, which is a good sign, that the model performs well,
    because the line it produces is closer to the points than they are close to each other if that makes sense
    """


if __name__ == "__main__":
    main()