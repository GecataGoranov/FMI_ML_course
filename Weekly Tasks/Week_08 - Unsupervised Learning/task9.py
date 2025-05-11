import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy


def main():
    data = pd.read_csv("../DATA/eurovision_voting.csv", index_col=0)
    
    mergings = hierarchy.linkage(data, method="single")
    hierarchy.dendrogram(mergings, labels=data.index, leaf_font_size=6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()