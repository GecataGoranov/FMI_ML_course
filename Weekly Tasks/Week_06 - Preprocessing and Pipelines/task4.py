import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay


def main():
    music_missing_vals = pd.read_json("../DATA/music_dirty_missing_vals.txt")
    music_missing_vals["genre"] = music_missing_vals["genre"].apply(lambda x: 1 if x == "Rock" else 0)

    X = music_missing_vals.drop(columns = ["genre"])
    y = music_missing_vals["genre"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 42)

    pipeline = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "median")),
            ("knn", KNeighborsClassifier(n_neighbors = 3))
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.tight_layout()
    plt.show()

    RocCurveDisplay.from_predictions(y_test, y_pred)
    plt.tight_layout()
    plt.show()

    """
    The model does not perform well at all. It's only slightly better than randomly guessing the data
    """

if __name__ == "__main__":
    main()