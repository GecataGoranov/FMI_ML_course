import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import ward_tree
from torch.utils.data import Dataset, DataLoader
from torch import is_tensor


class WaterDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = pd.read_csv(path)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        label = self.data.iloc[idx, -1]
        features = np.array(self.data.iloc[idx, :-1])

        return (features, label)
        

def main():
    water_dataset = WaterDataset("../DATA/water_train.csv")
    print("Number of instances:", len(water_dataset))
    print("Fifth item:", water_dataset[4])

    batches = DataLoader(water_dataset, batch_size=2, shuffle=True)
    train_features, train_labels = next(iter(batches))
    print(train_features, train_labels)


if __name__ == "__main__":
    main()