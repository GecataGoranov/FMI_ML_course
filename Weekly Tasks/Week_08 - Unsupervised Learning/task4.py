import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def main():
    data = pd.read_csv("../DATA/seeds_dataset.txt", sep="\t+", header=None, engine="python")
    data.columns = ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "varieties"]
    
    kmeans = KMeans(n_clusters=3, random_state=11)
    kmeans.fit(data)

    data["varieties"] = data["varieties"].astype(str)
    data["varieties"] = data["varieties"].apply(lambda x : "Canadian wheat" if x == "1" else "Kama wheat" if x == "2" else "Rosa wheat")

    df = pd.DataFrame({"labels":kmeans.labels_, "varieties":data["varieties"]})
    
    
    print(pd.crosstab(df["labels"], df["varieties"]))


if __name__ == "__main__":
    main()