import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


def main():
    data = pd.read_csv("../DATA/indian_liver_patient_dataset.csv", header=None)

    X_train, X_test, y_train, y_test = preprocess(data)

    bagging_classifier = BaggingClassifier(
        estimator=DecisionTreeClassifier(min_samples_leaf=8, random_state=1), 
        n_estimators=50,
        oob_score=True,
        random_state=1
    )
    bagging_classifier.fit(X_train, y_train)

    print("Mean accuracy of aggregator on OOB instances:", np.round(np.mean(bagging_classifier.oob_score_), 2))
    print("Test set accuracy:", np.round(bagging_classifier.score(X_test, y_test), 2))


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