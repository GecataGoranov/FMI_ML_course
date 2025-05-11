import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def main():
    churn_dataset = pd.read_csv("telecom_churn_clean.csv", index_col = 0)

    X = churn_dataset.drop(columns = ["churn"])
    y = churn_dataset["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

    print("Training Dataset Shape: ", X_train.shape)

    knn = KNeighborsClassifier(n_neighbors = 5)

    knn.fit(X_train, y_train)

    print("Accuracy when n_neighbors=5: ", round(knn.score(X_test, y_test), 4))

    train_accuracies = {}
    test_accuracies = {}

    iterations = 12

    for i in range(iterations):
        knn = KNeighborsClassifier(n_neighbors = i + 1)

        knn.fit(X_train, y_train)
        train_accuracies[i + 1] = round(knn.score(X_train, y_train), 4)
        test_accuracies[i + 1] = round(knn.score(X_test, y_test), 4)

    neighbors = np.array([i for i in range(1, 12)])
    print("neighbors=", neighbors)

    plt.plot(neighbors, [train_accuracies[i] for i in neighbors])
    plt.plot(neighbors, [test_accuracies[i] for i in neighbors])

    print("train_accuracies=", train_accuracies)
    print("test_accuracies=", test_accuracies)

    plt.show()


if __name__ == "__main__":
    main()