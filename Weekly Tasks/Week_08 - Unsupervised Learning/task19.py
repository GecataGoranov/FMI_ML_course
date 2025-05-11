import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


def main():
    data = pd.read_csv("../DATA/wikipedia-vectors.csv", index_col=0).T

    nmf = NMF(n_components=6)
    nmf.fit(data)
    data_transformed = nmf.transform(data)
    print(np.round(data_transformed[:6], 2))

    data_transformed = pd.DataFrame(data_transformed, index=data.T.columns)
    anne_denzel_nmf = data_transformed.loc[["Anne Hathaway", "Denzel Washington"]]

    print(anne_denzel_nmf)

    with open("../DATA/wikipedia-vocabulary-utf8.txt", "r", encoding="utf8") as f:
        columns = []
        for line in f:
            columns.append(line.replace("\n", ""))
        data.columns = columns

    print("The topic, that the articles about Anne Hathaway and Denzel Washington have in common, has the words:",
          pd.DataFrame(nmf.components_, columns=data.columns).loc[3].nlargest())

    


if __name__ == "__main__":
    main()