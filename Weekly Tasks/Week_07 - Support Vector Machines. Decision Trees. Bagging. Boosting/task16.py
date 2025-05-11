import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    gini_tree = DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=1)
    entropy_tree = DecisionTreeClassifier(max_depth=4, criterion="entropy", random_state=1)

    gini_tree.fit(X_train, y_train)
    entropy_tree.fit(X_train, y_train)

    print("Accuracy achieved by using entropy:", np.round(entropy_tree.score(X_test, y_test), 3))
    print("Accuracy achieved by using the gini index:", np.round(gini_tree.score(X_test, y_test), 3))

    """
    The results are swapped for me
    """


if __name__ == "__main__":
    main()