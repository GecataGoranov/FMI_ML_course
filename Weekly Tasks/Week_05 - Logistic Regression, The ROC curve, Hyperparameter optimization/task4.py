import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV


def main():
    diabetes_df = pd.read_csv("../DATA/diabetes_clean.csv")

    X = diabetes_df.drop(columns = ["glucose"])
    y = diabetes_df["glucose"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    param_grid = {"alpha" : np.linspace(0.00001, 1, 20)}

    cv = GridSearchCV(Lasso(random_state = 42), param_grid, cv = 6)
    cv.fit(X_train, y_train)

    print("Tuned lasso paramaters:", cv.best_params_)
    print("Tuned lasso score:", cv.best_score_)

    """ Using optimal hyperparameters does not guarantee a high performing model.
        Some models just aren't enough to handle some data."""
    
    # Again, the results are slightly different.


if __name__ == "__main__":
    main()