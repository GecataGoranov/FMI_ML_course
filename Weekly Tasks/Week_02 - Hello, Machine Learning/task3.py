import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def main():
    # Loading the dataset
    churn_dataset = pd.read_csv("telecom_churn_clean.csv")
    
    # Separating our X and y values
    X = churn_dataset[["account_length", "customer_service_calls"]]
    y = churn_dataset["churn"]

    # Loading our model and training it with the data provided
    knn_model = KNeighborsClassifier(n_neighbors = 6)
    knn_model.fit(X, y)

    # X_new from the assingment
    X_new = np.array([[30.0, 17.5],
                      [107.0, 24.1],
                      [213.0, 10.9]])

    # Finally, predicting 
    predictions = knn_model.predict(X_new)
    print(f"{predictions=}")

    """
    Добре, получих същите резултати като тези в условията, но останах озадачен... Доколкото  разбрах, трябва да ползваме само "account_length" и "customer_service_calls" като характеристики. Тогава как е възможно в X_new да има дробни числа?
    """


if __name__ == "__main__":
    main()