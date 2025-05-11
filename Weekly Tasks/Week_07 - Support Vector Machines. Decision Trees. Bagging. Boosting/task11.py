import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def main():
    X, y = load_digits(return_X_y=True)

    y_twos = [1 if y[i] == 2 else 0 for i in range(len(y))]

    param_grid = {
        "gamma" : [0.00001, 0.0001, 0.001, 0.01, 0.1]
    }
    svc = SVC()

    grid_search = GridSearchCV(svc, param_grid=param_grid)
    grid_search.fit(X, y_twos)

    print("Best CV parameters:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)


if __name__ == "__main__":
    main()