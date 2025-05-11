import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def main():
    digits, classes = load_digits(return_X_y = True, as_frame = True)
    print("Dataset shape:", digits.shape)
    print("Number of classes:", len(classes.unique()))

    X_train, X_test, y_train, y_test = train_test_split(digits, classes)

    # TODO: plot the images!!!

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    print("Training accuracy of logistic regression:", logreg.score(X_train, y_train))
    print("Validation accuracy of logistic regression:", logreg.score(X_test, y_test))

    svc = SVC()
    svc.fit(X_train, y_train)

    print("Training accuracy of non-linear support vector classifier:", svc.score(X_train, y_train))
    print("Validation accuracy of non-linear support vector classifier:", svc.score(X_test, y_test))

    """
    They both did a good job, but the SVC has a slight advantage,
    because it showed slightly better results on the test set.
    """


if __name__ == "__main__":
    main()