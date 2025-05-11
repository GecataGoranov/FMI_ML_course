from tkinter.tix import MAX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, Normalizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sympy import Max


def main():
    artists = pd.read_csv("../DATA/artists.csv", names=["artist_name"])
    scrobbler = pd.read_csv("../DATA/scrobbler-small-sample.csv")

    merged = pd.merge(artists, scrobbler, left_index=True, right_on="artist_offset")
    pivot = pd.pivot_table(merged, index="artist_name", columns="user_offset", values="playcount", fill_value=0)

    pipeline = Pipeline(steps=[
        ("scaler", MaxAbsScaler()),
        ("nmf", NMF(n_components=20)),
        ("normalizer", Normalizer())
    ])
    transformed_df = pipeline.fit_transform(pivot)

    cos_sims = cosine_similarity(transformed_df)[pivot.index.get_loc("Bruce Springsteen")]

    series = pd.Series(cos_sims, index=pivot.index).sort_values(ascending=False).head(5)
    print(series)

if __name__ == "__main__":
    main()