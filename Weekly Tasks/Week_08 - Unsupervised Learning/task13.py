import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv("../DATA/seeds_dataset.txt", sep="\t+", header=None, engine="python")
    data.columns = ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "varieties"]
    
    plt.scatter(x=data["width_of_kernel"], y=data["length_of_kernel"])
    
    plt.title(f"Pearson correlation: {np.round(data["width_of_kernel"].corr(data["length_of_kernel"]), 2)}")
    plt.xlabel("Width of kernel")
    plt.ylabel("Length of kernel")

    plt.show()


if __name__ == "__main__":
    main()