import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score


def main():
    data = pd.read_csv("../DATA/indian_liver_patient_dataset.csv", header=None)
    data.columns = ["age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp", "alb", "a/g_ratio", "has_liver_disease"]

    for col in data.columns:
        data.fillna({col : data[col].mode()[0]}, inplace=True)

    # Adjusting the target variable
    data["has_liver_disease"] -= 2
    data["has_liver_disease"] *= -1

    print(data, "\n")

    X = pd.get_dummies(data.drop(columns=["has_liver_disease"]), drop_first=True)
    y = data["has_liver_disease"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

    logreg = LogisticRegression(random_state=1)
    tree = DecisionTreeClassifier(min_samples_leaf=0.13 ,random_state=1)
    knn = KNeighborsClassifier(n_neighbors=27)

    logreg.fit(X_train, y_train)
    tree.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    logreg_y_pred = logreg.predict(X_test)
    tree_y_pred = tree.predict(X_test)
    knn_y_pred = knn.predict(X_test)

    print("Logistic Regression:", np.round(f1_score(y_test, logreg_y_pred), 3))
    print("K Nearest Neighbours:", np.round(f1_score(y_test, knn_y_pred), 3))
    print("Classification Tree:", np.round(f1_score(y_test, tree_y_pred), 3))

    voting_classifier = VotingClassifier(
        estimators=[
            ("logreg", logreg),
            ("knn", knn),
            ("tree", tree)
    ])
    voting_classifier.fit(X_train, y_train)

    voting_y_pred = voting_classifier.predict(X_test)
    print("\nVoting Classifier:", np.round(f1_score(y_test, voting_y_pred), 3))


if __name__ == "__main__":
    main()