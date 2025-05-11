from http.cookies import SimpleCookie
from multiprocessing import Pipe
from tkinter import Grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def main():
    music_df = pd.read_json("../DATA/music_dirty_missing_vals.txt")
    music_df["genre"] = music_df["genre"].apply(lambda x: 1 if x == "Rock" else 0)

    X = music_df.drop(columns = ["genre"])
    y = music_df["genre"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 42)

    pipeline = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(random_state = 42))
        ]
    )

    param_grid = {
        "logreg__solver" : ["newton-cg", "saga", "lbfgs"],
        "logreg__C" : np.linspace(0.001, 1.0, 10)
    }

    grid_search = GridSearchCV(pipeline, param_grid = param_grid, scoring = "accuracy")

    grid_search.fit(X_train, y_train)

    print("Tuned Logistic Regression Parameters:", grid_search.best_params_)
    print("Accuracy:", grid_search.score(X_test, y_test))



if __name__ == "__main__":
    main()