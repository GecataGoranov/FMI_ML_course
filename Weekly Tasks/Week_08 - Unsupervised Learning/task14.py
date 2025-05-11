import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def main():
    data = pd.read_csv("../DATA/seeds_dataset.txt", sep="\t+", header=None, engine="python")
    data.columns = ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "varieties"]
    
    pca = PCA(n_components=2)
    pca.fit(data[["width_of_kernel", "length_of_kernel"]])

    data_new = pca.transform(data[["width_of_kernel", "length_of_kernel"]])

    plt.scatter(x=data_new[:,0], y=data_new[:,1])
    
    plt.title(f"Pearson correlation: {np.round(np.corrcoef(data_new[:,0], data_new[:,1])[0,1], 2)}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plt.show()


if __name__ == "__main__":
    main()