import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    music_df = pd.read_csv("../DATA/music_clean.csv", index_col=0)

    X = music_df.drop(columns = ["genre"])
    y = music_df["genre"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 21)

    param_grid = {
        "logistic_regression__C" : np.linspace(0.001, 1.0, 20)
    }

    pipeline_without_scaling = Pipeline(
        steps = [
            ("logistic_regression", LogisticRegression(random_state = 21))
        ]        
    )

    pipeline_with_scaling = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("logistic_regression", LogisticRegression(random_state = 21))
        ]        
    )

    grid_search_without_scaling = GridSearchCV(estimator = pipeline_without_scaling, param_grid = param_grid)
    grid_search_without_scaling.fit(X_train, y_train)

    print("Without scaling:", grid_search_without_scaling.score(X_test, y_test))
    print("Without scaling:", grid_search_without_scaling.best_params_)

    grid_search_with_scaling = GridSearchCV(estimator = pipeline_with_scaling, param_grid = param_grid)
    grid_search_with_scaling.fit(X_train, y_train)

    print("With scaling:", grid_search_with_scaling.score(X_test, y_test))
    print("With scaling:", grid_search_with_scaling.best_params_)

    """
    Again, scaling significantly improves the model's performance and this time it actually performs pretty well.
    """


if __name__ == "__main__":
    main()