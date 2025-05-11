import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import NMF


def main():
    data = pd.read_csv("../DATA/wikipedia-vectors.csv", index_col=0).T

    nmf = NMF(n_components=6)
    nmf.fit(data)

    data_transformed = nmf.transform(data)

    normalizer = Normalizer()
    data_normalized = normalizer.fit_transform(data_transformed)

    cos_sims = cosine_similarity(data_normalized)[data.index.get_loc("Cristiano Ronaldo")]

    series = pd.Series(cos_sims, index=data.index).sort_values(ascending=False).head(5)
    print(series)


if __name__ == "__main__":
    main()