import numpy as np
import pandas as pd


def create_sequences(data, seq_length):
    sequences = []
    targets = []

    for i in range(data.shape[0] - seq_length - 1):
        sequences.append(data[i:i+seq_length])
        targets.append(data.iloc[i+seq_length])

    return np.array(sequences), np.array(targets)


def main():
    data = []
    j = 0
    for i in range(101):
        data.append(np.linspace(j, j+4, 5))
        j += 1
    
    data = pd.DataFrame(data).astype(int)
    seq, tars = create_sequences(data, 5)

    print("First five training examples:", seq[0])
    print("First five target values:", tars[0])

    electricity_train = pd.read_csv("../DATA/electricity_consumption/electricity_train.csv")
    X_train, y_train = create_sequences(electricity_train, 5)

    # print(X_train)

    print(f"X_train.shape={X_train.shape}")
    print(f"y_train.shape={y_train.shape}")
    print("Length of training TensorDataset:", y_train.shape[0])


if __name__ == "__main__":
    main()