import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
from sympy import rotations


def main():
    music_df = pd.read_json("../DATA/music_dirty.txt")

    plt.boxplot([music_df[music_df["genre"] == genre]["popularity"] for genre in sorted(music_df["genre"].unique())],
                tick_labels = sorted(music_df["genre"].unique()),
                patch_artist = True,
                boxprops=dict(facecolor = "blue"),
                medianprops=dict(color = "green"),
                whiskerprops=dict(color = "blue"))
    
    plt.title("Boxplot grouped by genre")

    plt.xlabel("genre")
    plt.xticks(rotation = 45)

    plt.ylabel("popularity")
    
    print("Shape before one-hot-encoding:", music_df.shape)
    dummies = pd.get_dummies(music_df["genre"], drop_first = True)

    everything_df = pd.concat([music_df.drop(columns = ["genre"]), dummies], axis = 1)
    print("Shape after one-hot-encoding:", everything_df.shape)

    plt.show()


if __name__ == "__main__":
    main()