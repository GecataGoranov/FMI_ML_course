import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score


def main():
    data = pd.read_csv("../DATA/indian_liver_patient_dataset.csv", header=None)

    X_train, X_test, y_train, y_test = preprocess(data)
    
    bagging_classifier = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=1), 
        n_estimators=50,
        random_state=1
    )
    tree = DecisionTreeClassifier(random_state=1)

    bagging_classifier.fit(X_train, y_train)
    tree.fit(X_train, y_train)

    bagging_y_pred = bagging_classifier.predict(X_test)
    tree_y_pred = tree.predict(X_test)

    print("Test set f1-score of aggregator:", np.round(f1_score(y_test, bagging_y_pred), 2))
    print("Test set f1-score of single decision tree:", np.round(f1_score(y_test, tree_y_pred), 2))

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