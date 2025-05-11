from os import pipe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


def main():
    data = pd.read_csv("../DATA/wikipedia-vectors.csv", index_col=0).T
    
    pipeline = Pipeline(steps=[
        ("svd", TruncatedSVD(n_components=50)),
        ("kmeans", KMeans(n_clusters=6))
    ])
    labels = pipeline.fit_predict(data)
    labels_article_df = pd.DataFrame({"label":labels, "article":data.T.columns}).sort_values(by="label")
    print(labels_article_df)


if __name__ == "__main__":
    main()