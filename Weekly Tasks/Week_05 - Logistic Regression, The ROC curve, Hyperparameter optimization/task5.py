import numpy as np
import pandas as pd

import scipy as sp
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.linear_model import LogisticRegression


def main():
    diabetes_df = pd.read_csv("../DATA/diabetes_clean.csv")

    kf = KFold(n_splits = 6, shuffle = True, random_state = 42)

    X_train, X_test, y_train, y_test = train_test_split(diabetes_df.drop(columns = ["diabetes"]), diabetes_df["diabetes"], stratify = diabetes_df["diabetes"], test_size = 0.3, random_state = 42)

    param_distributions = {
        "penalty" : ["l1", "l2"],
        "tol" : np.linspace(0.00001, 1.0, 50),
        "C" : np.linspace(0.1, 1.0, 50),
        "class_weight" : ["balanced", {0 : 0.8, 1 : 0.2}]
    }

    random_search = RandomizedSearchCV(
        param_distributions = param_distributions,
        estimator = LogisticRegression(random_state = 42),
        random_state = 42,
        cv = kf
    )

    random_search.fit(X_train, y_train)

    print("Tuned Logistic Regression Parameters:", random_search.best_params_)
    print("Tuned Logistic Regression Best Accuracy Score:", random_search.best_score_)


if __name__ == "__main__":
    main()