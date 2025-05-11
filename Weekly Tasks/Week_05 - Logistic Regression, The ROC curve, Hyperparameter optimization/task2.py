import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay


def main():
    diabetes_df = pd.read_csv("../DATA/diabetes_clean.csv")
    
    X = diabetes_df.drop(columns = ["diabetes"])
    y = diabetes_df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 42)

    logistic_regression = LogisticRegression(random_state = 42)
    logistic_regression.fit(X_train, y_train)

    y_pred = logistic_regression.predict_proba(X_test)[:,1]

    RocCurveDisplay.from_predictions(y_test, y_pred)
    plt.tight_layout()
    plt.show()

    # The correct answer is C


if __name__ == "__main__":
    main()