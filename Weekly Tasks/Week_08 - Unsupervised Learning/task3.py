import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def main():
    data = pd.read_csv("../DATA/seeds_dataset.txt", sep="\t+", header=None)
    data.columns = ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "varieties"]
    print(data)

    inertias = np.ndarray(shape=(6,2))

    for i in range(1, 7):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias[i-1] = [i, kmeans.inertia_]

    plt.plot(inertias[:, 0], inertias[:, 1])
    plt.scatter(x=inertias[:, 0], y=inertias[:, 1])

    plt.title("Inertia per number of clusters")
    plt.xlabel("number of clusters, k")
    plt.ylabel("Inertia")

    plt.show()

    """
    In this case, 3 clusters looks like the best decision, because after that, the inertia doesn't decrease as quickly.
    """


if __name__ == "__main__":
    main()