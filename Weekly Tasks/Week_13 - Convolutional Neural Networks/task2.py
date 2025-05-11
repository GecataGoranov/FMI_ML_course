import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import is_tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class CloudsDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.img_labels = sorted([r"cirriform clouds", r"clear sky", r"cumulonimbus clouds",
                           r"cumulus clouds", r"high cumuliform clouds",
                           r"stratiform clouds", r"stratocumulus clouds"])
        self.labels_to_idx = {lab : idx for idx, lab in enumerate(self.img_labels)}
        
        images_list = []
        for label in self.img_labels:
            for img in os.listdir(path+label):
                img_path = path + label + "/" + img
                images_list.append((img_path, self.labels_to_idx[label]))

        self.images_ds = pd.DataFrame(images_list)
            
        self.transform = transform

    def __len__(self):
        return self.images_ds.shape[0]
    
    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.to_list()

        label = torch.tensor(self.images_ds.iloc[idx, 1])
        img = PIL.Image.open(str(self.images_ds.iloc[idx, 0]))

        if self.transform:
            img = self.transform(img)

        return (img, label)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.mp = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.ff = nn.Linear(64 * 64 * 16, 7)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.mp(x)
        x = self.flatten(x)
        x = F.log_softmax(self.ff(x))

        return x


def train_model(dataloader_train, optimizer, net, num_epochs, create_plot=False):
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        i = 1
        for img, labels in tqdm(dataloader_train, f"Epoch {i}"):
            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, labels)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            i += 1
        
        losses.append(np.mean(epoch_losses))
        print("Average training loss per batch:", np.mean(epoch_losses))

    print("Average training loss per epoch:", np.mean(losses))

    if(create_plot):
        plt.plot(losses)

        plt.title("Loss per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.show()


def main():
    path = "../DATA/clouds/clouds_train/"
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((0, 45)),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])
    clouds_train = CloudsDataset(path, train_transforms)
    
    first_image = clouds_train[0][0]
    plt.imshow(first_image.permute(1,2,0))
    plt.tight_layout()
    plt.show()
    
    dataloader_train = DataLoader(clouds_train, batch_size=16, shuffle=True)
    net = Net()
    adamW = optim.AdamW(net.parameters(), lr=0.001)

    train_model(dataloader_train, adamW, net, 20, True)



if __name__ == "__main__":
    main()