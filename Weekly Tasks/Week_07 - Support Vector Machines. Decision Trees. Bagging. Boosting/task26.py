import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


def main():
    data = pd.read_csv("../DATA/indian_liver_patient_dataset.csv", header=None)

    X_train, X_test, y_train, y_test = preprocess(data)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    tree = DecisionTreeClassifier(random_state=1)
    param_grid = {
        "max_depth" : [2, 3, 4],
        "min_samples_leaf" : [0.12, 0.14, 0.16, 0.18]
    }
    grid_search = GridSearchCV(
        estimator=tree,
        param_grid=param_grid,
        cv=kf,
        scoring="roc_auc"
    )
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    print("Test set ROC AUC:", grid_search.score(X_test, y_test))

    """
    WHY?
    """


def preprocess(dataset : pd.DataFrame):
    """
    Function, that takes a dataset, adds column names,
    adjusts the target variable to be 0 and 1,
    one-hot encodes the categorical features and
    splits the data into train and test set

    Parameters
    ----------
    dataset: pandas DataFrame of the whole dataset
    """

    dataset.columns = ["age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp", "alb", "a/g_ratio", "has_liver_disease"]

    for col in dataset.columns:
        dataset.fillna({col : dataset[col].mode()[0]}, inplace=True)

    # Adjusting the target variable
    dataset["has_liver_disease"] -= 2
    dataset["has_liver_disease"] *= -1

    X = pd.get_dummies(dataset.drop(columns=["has_liver_disease"]), drop_first=True)
    y = dataset["has_liver_disease"]

    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)


if __name__ == "__main__":
    main()