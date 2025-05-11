import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import is_tensor
from torchmetrics.classification import F1Score, Accuracy


torch.manual_seed(42)

class WaterDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = pd.read_csv(path)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        label = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float32)
        features = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)

        return (features, label)
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")

        self.fc2 = nn.Linear(16, 32)
        init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")

        self.fc3 = nn.Linear(32, 16)
        init.kaiming_uniform_(self.fc3.weight, nonlinearity="relu")

        self.fc4 = nn.Linear(16, 8)
        init.kaiming_uniform_(self.fc4.weight, nonlinearity="relu")

        self.fc5 = nn.Linear(8, 1)
        init.kaiming_uniform_(self.fc5.weight, nonlinearity="sigmoid")


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))

        return x
    

def train_model(dataloader_train, optimizer, net, num_epochs, create_plot=False):
    criterion = nn.BCELoss()
    losses = []

    for epoch in tqdm(range(num_epochs)):
        epoch_losses = []
        for features, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = net(features)
            labels = labels.view(-1, 1)
            loss = criterion(outputs, labels)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        losses.append(np.mean(epoch_losses))
    print("Average loss:", np.mean(losses))

    if(create_plot):
        plt.plot(losses)

        plt.title("Loss per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.show()
    
            
def main():
    net = Net()

    dataset_train = WaterDataset("../DATA/water_train.csv")
    dataset_test = WaterDataset("../DATA/water_test.csv")

    dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=True)

    adamW = optim.AdamW(net.parameters(), lr=0.0005)

    train_model(dataloader_train, adamW, net, 8000, create_plot=True)

    f1 = F1Score(task="binary")
    acc = Accuracy(task="binary")

    net.eval()
    with torch.no_grad():
        for features, labels in dataloader_test:
            outputs = net(features)
            preds = (outputs >= 0.5).float()
            f1.update(preds, labels.view(-1, 1))
            acc.update(preds, labels.view(-1, 1))

    f1__score = f1.compute()
    accuracy = acc.compute()

    print("F1 score on test set:", f1__score.item())
    print("Accuracy on test set:", accuracy.item())


if __name__ == "__main__":
    main()