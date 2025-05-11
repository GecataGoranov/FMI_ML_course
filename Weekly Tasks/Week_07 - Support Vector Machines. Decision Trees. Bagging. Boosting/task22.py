import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def main():
    data = pd.read_csv("../DATA/bike_sharing.csv")

    # For some reason it gave me a slightly better score without drop_first=True
    data_ohe = pd.get_dummies(data, columns=["season", "weather"])

    # Also dropping the "datetime" column, because the model still couldn't fit
    # the data even after converting it to "datetime"
    X = data_ohe.drop(columns=["count", "datetime"])
    y = data_ohe["count"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    random_forest = RandomForestRegressor(n_estimators=25, random_state=2)
    random_forest.fit(X_train, y_train)

    y_pred = random_forest.predict(X_test)
    print("Test set RMSE:", np.round(root_mean_squared_error(y_test, y_pred), 2))

    feature_importances_series = pd.Series(random_forest.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    sns.barplot(y=feature_importances_series.index, x=feature_importances_series, color="lime")
    plt.show()


if __name__ == "__main__":
    main()