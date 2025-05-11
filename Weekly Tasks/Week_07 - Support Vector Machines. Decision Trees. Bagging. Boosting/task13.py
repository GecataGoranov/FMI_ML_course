import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier


def main():
    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    y_twos_train = [1 if y_train[i] == 2 else 0 for i in range(len(y_train))]
    y_twos_test = [1 if y_test[i] == 2 else 0 for i in range(len(y_test))]

    param_grid = {
        "alpha" : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
        "loss" : ["hinge", "log_loss"]
    }

    sgd = SGDClassifier(random_state=0)

    grid_search = GridSearchCV(sgd, param_grid=param_grid)
    grid_search.fit(X_train, y_twos_train)

    print("Best CV params:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)
    print("Test accuracy of best grid search hypers:", grid_search.score(X_test, y_twos_test))

    """
    I couldn't achieve the same resulst for some reason
    """


if __name__ == "__main__":
    main()