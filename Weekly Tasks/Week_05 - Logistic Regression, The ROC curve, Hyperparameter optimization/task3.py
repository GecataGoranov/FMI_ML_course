from turtle import title
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def main():
    diabetes_df = pd.read_csv("../DATA/diabetes_clean.csv")
    
    X = diabetes_df.drop(columns = ["diabetes"])
    y = diabetes_df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 42)

    knn = KNeighborsClassifier()
    logistic_regression = LogisticRegression(random_state = 42)

    knn.fit(X_train, y_train)
    print("Model KNN trained!")

    logistic_regression.fit(X_train, y_train)
    print("Model LogisticRegression trained!")

    knn_proba = knn.predict_proba(X_test)
    knn_pred = knn.predict(X_test)

    print("KNN AUC:", roc_auc_score(y_test, knn_proba[:, 1]))
    print("KNN Metrics:\n", classification_report(y_test, knn_pred))

    logreg_proba = logistic_regression.predict_proba(X_test)
    logreg_pred = logistic_regression.predict(X_test)

    print("LogisticRegression AUC:", roc_auc_score(y_test, logreg_proba[:, 1]))
    print("LogisticRegression Metrics:\n", classification_report(y_test, logreg_pred))

    knn_cm = confusion_matrix(y_test, knn_pred)
    logreg_cm = confusion_matrix(y_test, logreg_pred)

    fig, axes = plt.subplots(1, 2, sharey = "row")

    disp1 = ConfusionMatrixDisplay(knn_cm)
    disp1.plot(ax = axes[0])
    disp1.ax_.set_title("KNN")
    disp1.im_.colorbar.remove()

    disp2 = ConfusionMatrixDisplay(logreg_cm)
    disp2.plot(ax = axes[1])
    disp2.ax_.set_title("Logistic Regression")
    disp2.im_.colorbar.remove()

    fig.colorbar(disp2.im_, ax = axes)

    plt.show()

    """ Judging by our results, the Logistic Regression performs
    slightly better than KNN, but the difference is not significant.
    That becomes apparent when comparing their scores. """


if __name__ == "__main__":
    main()