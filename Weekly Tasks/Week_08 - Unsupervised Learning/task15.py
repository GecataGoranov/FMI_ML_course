import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def main():
    data = pd.read_csv("../DATA/seeds_dataset.txt", sep="\t+", header=None, engine="python")
    data.columns = ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "varieties"]
    
    pca = PCA(n_components=2)
    pca.fit(data[["width_of_kernel", "length_of_kernel"]])

    print(pca.singular_values_)

    plt.scatter(x=data["width_of_kernel"], y=data["length_of_kernel"])
    plt.arrow(
        x=data["width_of_kernel"].mean(),
        y=data["length_of_kernel"].mean(), 
        dx=3 * pca.get_covariance()[0,0], 
        dy=3 * pca.get_covariance()[0,1],
        width=0.01,
        head_width = 0.05,
        color="red",
        hatch="*"
        )


    plt.title(f"Pearson correlation: {np.round(data["width_of_kernel"].corr(data["length_of_kernel"]), 2)}")
    plt.xlabel("Width of kernel")
    plt.ylabel("Length of kernel")

    plt.show()


if __name__ == "__main__":
    main()