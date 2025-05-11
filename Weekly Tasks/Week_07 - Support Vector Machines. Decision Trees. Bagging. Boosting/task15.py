import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlxtend.plotting import plot_decision_regions


def plot_labeled_decision_regions(X,y, models):    
    '''
    Function producing a scatter plot of the instances contained 
    in the 2D dataset (X,y) along with the decision 
    regions of two trained classification models contained in the
    list 'models'.
            
    Parameters
    ----------
    X: pandas DataFrame corresponding to two numerical features 
    y: pandas Series corresponding the class labels
    models: list containing two trained classifiers 
    '''
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for i, model in enumerate(models):
        plot_decision_regions(X.values,y.values, model, legend= 2, ax = ax[i])
        ax[i].set_title(model.__class__.__name__)
        ax[i].set_xlabel(X.columns[0])
        if i == 0:
            ax[i].set_ylabel(X.columns[1])
        ax[i].set_ylim(X.values[:,1].min(), X.values[:,1].max())
        ax[i].set_xlim(X.values[:,0].min(), X.values[:,0].max())
    plt.tight_layout()
    plt.show()


def main():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X[["mean radius", "mean concave points"]], y, test_size=0.2, stratify=y, random_state=1)

    tree = DecisionTreeClassifier(max_depth=6, random_state=1)
    tree.fit(X_train, y_train)

    logreg = LogisticRegression(random_state=1)
    logreg.fit(X_train, y_train)

    plot_labeled_decision_regions(X_test, y_test, [logreg, tree])
    plt.show()

    """
    The Logistic Regression just draws a straight line between the two classes and calls it a day.
    The Decision Tree creates some sort of "regions" instead.
    """


if __name__ == "__main__":
    main()