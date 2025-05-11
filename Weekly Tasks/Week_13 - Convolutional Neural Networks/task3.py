import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from torch import is_tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics import Precision, Recall, F1Score
from tqdm import tqdm
from pprint import pp


torch.manual_seed(42)


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
        
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((64, 64))
            ])

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
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    losses = []
    i = 1

    for epoch in range(num_epochs):
        epoch_losses = []
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

    print("\nSummary statistics:")
    print("Average training loss per epoch:", np.mean(losses))
    end_time = time.time()

    if(create_plot):
        plt.plot(losses)

        plt.title("Loss per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.show()

    return end_time - start_time


def main():
    train_path = "../DATA/clouds/clouds_train/"
    test_path = "../DATA/clouds/clouds_test/"

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((0, 45)),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])

    clouds_train = CloudsDataset(train_path, train_transforms)
    clouds_test = CloudsDataset(test_path)
    
    dataloader_train = DataLoader(clouds_train, batch_size=16, shuffle=True)
    dataloader_test = DataLoader(clouds_test, batch_size=1, shuffle=True)

    net = Net()
    adamW = optim.AdamW(net.parameters(), lr=0.001)

    time_taken = train_model(dataloader_train, adamW, net, 20, True)

    precision_macro = Precision(task="multiclass", average="macro", num_classes=7)
    recall_macro = Recall(task="multiclass", average="macro", num_classes=7)
    f1_macro = F1Score(task="multiclass", average="macro", num_classes=7)
    f1_binary = [F1Score(task="binary") for _ in range(len(clouds_test.img_labels))]

    net.eval()
    with torch.no_grad():
        for img, labels in dataloader_test:
            outputs = net(img)
            probs = F.softmax(outputs, dim=1)
            precision_macro.update(outputs, labels)
            recall_macro.update(outputs, labels)
            f1_macro.update(outputs, labels)

            for i in range(7):
                binary_targets = (labels == i).int()  # Convert labels into binary (1 for class `i`, 0 otherwise)
                f1_binary[i].update(probs[:, i], binary_targets)

    precision = precision_macro.compute()
    recall = recall_macro.compute()
    f1 = f1_macro.compute()
    f1_binary_scores = {label : metric.compute().item() for label, metric in zip(clouds_test.img_labels, f1_binary)}

    print("Precision:", precision.item())
    print("Recall:", recall.item())
    print("F1:", f1.item())
    print("Total time taken to train the model in seconds:", time_taken)
    print("\nPer class F1 score")
    pp(f1_binary_scores)


    """
    I'm not really sure if I've done the metrics correctly
    """


if __name__ == "__main__":
    main()