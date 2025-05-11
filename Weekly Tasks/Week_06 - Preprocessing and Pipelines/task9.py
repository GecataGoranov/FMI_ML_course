import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

def main():
    music_df = pd.read_csv("../DATA/music_clean.csv", index_col = 0)

    music_df["popularity"] = np.where(music_df["popularity"] >= np.median(music_df["popularity"]), 1, 0)

    X = music_df.drop(columns = ["popularity"])
    y = music_df["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 42)

    kf = KFold(n_splits = 6, shuffle = True, random_state = 12)

    logreg_pipeline = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression())
        ]
    )

    knn_pipeline = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier())
        ]
    )

    tree_pipeline = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("tree", DecisionTreeClassifier())
        ]
    )

    logreg_crossval = cross_validate(logreg_pipeline, X_train, y_train, cv = kf)
    knn_crossval = cross_validate(knn_pipeline, X_train, y_train, cv = kf)
    tree_crossval = cross_validate(tree_pipeline, X_train, y_train, cv = kf)

    plt.boxplot([logreg_crossval["test_score"], knn_crossval["test_score"], tree_crossval["test_score"]],
                tick_labels = ["Logistic Regression", "KNN", "Decision Tree Classifier"])
    plt.show()


if __name__ == "__main__":
    main()