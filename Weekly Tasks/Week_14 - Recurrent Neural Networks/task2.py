import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics import MeanSquaredError
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=2,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )

        self.ff = nn.Linear(32, 1)
    
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32)
        out, h_n = self.rnn(x, h0)
        out = self.ff(out)
        return out
    

def create_sequences(data, seq_length):
    sequences = []
    targets = []

    for i in range(data.shape[0] - seq_length - 1):
        sequences.append(data[i:i+seq_length])
        targets.append(data.iloc[i+seq_length])

    return np.array(sequences), np.array(targets)

    

def train_model(dataloader_train, optimizer, net, num_epochs, create_plot=False):
    criterion = nn.MSELoss()
    losses = []
    i = 1

    for epoch in range(num_epochs):
        epoch_losses = []
        for sequences, target in tqdm(dataloader_train, f"Epoch: {i}"):
            optimizer.zero_grad()
            outputs = net(sequences)
            loss = criterion(outputs, target)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        i += 1
        losses.append(np.mean(epoch_losses))
        print("Average MSE loss:", np.mean(epoch_losses))


def main():
    data = pd.read_csv("../DATA/electricity_consumption/electricity_train.csv")
    X_train, y_train = create_sequences(data, 5)

    dataset_train = TensorDataset(
        torch.from_numpy(X_train[]).float(),
        torch.from_numpy(y_train).float()
    )
    dataloader_train = DataLoader(dataset_train, batch_size=32)

    net = Net()
    optimizer = optim.AdamW(net.parameters(), lr=0.0001)

    train_model(dataloader_train, optimizer, net, 3, True)




if __name__ == "__main__":
    main()