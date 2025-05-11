import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import NMF, PCA
from sympy import comp
from zmq import NORM_SEGMENT_SIZE


def plot_first_digit(row: np.array) -> None:
    plt.imshow(row.reshape(13, 8), cmap='gray')
    plt.title("First Digit")
    plt.show()

def main():
    images = pd.read_csv("../DATA/lcd-digits.csv", header=None).to_numpy()
    plot_first_digit(images[0])

    nmf = NMF(n_components=7)
    nmf.fit(images)

    fig, axes = plt.subplots(nrows=2, ncols=4)
    plt.suptitle("Features Learned by NMF")

    axes = axes.flatten()
    i = 0
    for component in nmf.components_:
        axes[i].imshow(component.reshape(13,8), cmap="gray")
        axes[i].axis("off")
        i += 1

    axes[7].axis("off")

    plt.show()

    pca = PCA(n_components=7)
    pca.fit(images)

    fig, axes = plt.subplots(nrows=2, ncols=4)

    plt.suptitle("Features learned by PCA")
    
    axes = axes.flatten()
    i = 0
    for component in pca.components_:
        axes[i].imshow(component.reshape(13,8), cmap="gray")
        axes[i].axis("off")
        i += 1
    axes[7].axis("off")

    plt.show()


if __name__ == "__main__":
    main()