import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import manifold



def main():
    data = pd.read_csv("../DATA/seeds_dataset.txt", sep="\t+", header=None, engine="python")
    data.columns = ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "varieties"]
    
    learning_rates = [20, 50, 100, 200, 1000]

    for rate in learning_rates:
        tsne = manifold.TSNE(learning_rate=rate)
        transformed = tsne.fit_transform(data)

        plt.scatter(transformed[:,0], transformed[:,1], c=data["varieties"])
        plt.show()


if __name__ == "__main__":
    main()