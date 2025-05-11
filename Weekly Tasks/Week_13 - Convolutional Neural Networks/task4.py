import PIL
from jupyter_client import KernelConnectionInfo
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


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


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
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.mp3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.mp4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.mp5 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.ff1 = nn.Linear(25088, 4096)
        self.ff2 = nn.Linear(4096, 7)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.bn1(x)
        x = self.mp1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.bn2(x)
        x = self.mp2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.bn3(x)
        x = self.mp3(x)

        x = F.relu(self.conv4_1(x))
        x = self.bn4(x)
        x = self.mp4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.bn5(x)
        x = self.mp5(x)

        x = self.flatten(x)
        x = F.relu(self.ff1(x))
        x = self.ff2(x)

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
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    clouds_train = CloudsDataset(train_path, train_transforms)
    clouds_test = CloudsDataset(test_path)
    
    dataloader_train = DataLoader(clouds_train, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(clouds_test, batch_size=1, shuffle=True)

    net = Net()
    net.apply(initialize_weights)
    adamW = optim.AdamW(net.parameters(), lr=0.005)

    time_taken = train_model(dataloader_train, adamW, net, 120, True)

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
                binary_targets = (labels == i).int()
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
    For this task I replicated the VGG-16 architecture. Because it's too heavy for my laptop,
    I ran it on Colab and here are the metrics I got:

    Epoch 120 - Avg Loss: 0.2328773383051157

    Precision: 0.7810518741607666
    Recall: 0.806952953338623
    F1: 0.7851810455322266

    Per class F1 score
        {'cirriform clouds': 0.7067669034004211,
        'clear sky': 0.9770992398262024,
        'cumulonimbus clouds': 0.5925925970077515,
        'cumulus clouds': 0.8034934401512146,
        'high cumuliform clouds': 0.790243923664093,
        'stratiform clouds': 0.9210526347160339,
        'stratocumulus clouds': 0.7239263653755188}
    """


if __name__ == "__main__":
    main()