import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main():
    music_clean = pd.read_csv("../DATA/music_clean.csv", index_col = 0)

    print(music_clean.head(5))
    
    X = music_clean.drop(columns = ["loudness"])
    y = music_clean["loudness"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    pipeline_without_scaling = Pipeline(
        steps = [
            ("lasso", Lasso(alpha = 0.5, random_state = 42))
        ]
    )

    pipeline_with_scaling = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha = 0.5, random_state = 42))
        ]
    )

    pipeline_without_scaling.fit(X_train, y_train)
    print("Without scaling:", pipeline_without_scaling.score(X_test, y_test))

    pipeline_with_scaling.fit(X_train, y_train)
    print("With scaling:", pipeline_with_scaling.score(X_test, y_test))

    """
    The model without scaling performs very bad.
    The model with scaling performs significantly better, although it's also not very good.
    """


if __name__ == "__main__":
    main()