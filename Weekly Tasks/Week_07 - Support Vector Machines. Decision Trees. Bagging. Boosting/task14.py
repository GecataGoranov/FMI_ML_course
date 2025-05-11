import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X[:, [0, 7]], y, test_size=0.2, stratify=y, random_state=1)

    tree = DecisionTreeClassifier(max_depth=6, random_state=1)
    tree.fit(X_train, y_train)

    print("First 5 predictions:", tree.predict(X_test)[:5])
    print("Test set accuracy:", np.round(tree.score(X_test, y_test), 2))


if __name__ == "__main__":
    main()