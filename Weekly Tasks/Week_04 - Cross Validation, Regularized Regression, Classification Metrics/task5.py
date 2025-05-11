import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def main():
    diabetes_df = pd.read_csv("../DATA/diabetes_clean.csv")

    X_train, X_test, y_train, y_test = train_test_split(diabetes_df[["bmi", "age"]], diabetes_df["diabetes"], test_size = 0.3, stratify = diabetes_df["diabetes"], random_state = 42)

    knn = KNeighborsClassifier(n_neighbors = 6)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("classification report:\n", classification_report(y_test, y_pred, target_names = ["No diabetes", "Diabetes"]))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

    plt.show()

    # 20 true positives were predicted
    # 16 false positives were predicted
    # The f1-score for the "No diabetes" class is higher




if __name__ == "__main__":
    main()