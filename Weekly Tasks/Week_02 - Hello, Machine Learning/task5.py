import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


class KNN:
    # Конструктор, запазващ n_neighbors
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    # Функцията, изчисляваща разстоянията между отделните точки
    def __calculate_dist(self, u, v):
        sum_square_diffs = sum([(u.iloc[i] - v.iloc[i]) ** 2 for i in range(len(u))])
        return math.sqrt(sum_square_diffs.sum())

    # На fit му се налага само да запомни точките от тренировъчните данни
    def fit(self, X, y):
        self.__X = X
        self.__y = y

    def predict(self, X):
        # Създаваме си DataFrame, където да пазим разстоянията между всяка точка
        distances = pd.DataFrame(index = [i for i in X.index], columns = [j for j in self.__X.index])

        # Създаваме си DataFrame, в който да пазим най-малките n_neighbors на брой точки за всяка от данните, които трябва да predict-нем
        nsmallest_indexes = pd.DataFrame(index = [i for i in X.index], columns = [j for j in range(self.n_neighbors)])

        predictions = []
        
        for i in X.index:
            for j in self.__X.index:
                # Пълним таблицата с разстояния
                distances.loc[i, j] = self.__calculate_dist(X.iloc[lambda x: x.index == i], self.__X.iloc[lambda x: x.index == j])

            # Взимаме индексите на най-близките n_neighbors точки и ги запазваме в таблицата с индекси
            neighbors_indexes = distances.loc[i].astype(float).nsmallest(self.n_neighbors).index
            nsmallest_indexes.loc[i] = neighbors_indexes

            # Създаваме Series обект и го пълним с класовете на най-близките точки
            classes_list = []
            for idx in neighbors_indexes:
                classes_list.append(self.__y.loc[idx])

            classes_series = pd.Series(classes_list)

            # Добавяме най-често срещания клас в predictions. Тъй като елементите са наредени в ред на най-голяма близост, мога да си позволя да го оставя така дори и да има елементи с равен брой съседи. То все пак ще върне класа, от който има най-близък съсед.
            predictions.append(classes_series.value_counts().idxmax())

        return predictions
    
    # Пускаме predictions и смятаме съотношението на познатите към всички класове
    def score(self, X, y):
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        predictions = pd.DataFrame(self.predict(X))
        accurate = 0

        for i in predictions.index:
            if(predictions.iloc[i].values == y.iloc[i].values):
                accurate += 1

        return accurate / len(y)


def main():
    # Избрах си същия източник на данни като в останалите задачи
    churn_df = pd.read_csv("telecom_churn_clean.csv", index_col = 0)

    X = churn_df.drop(columns = ["churn"])
    y = churn_df["churn"]

    # Сложих random state, да да съм сигурен, че ще се получат същите резултати, които видях и аз
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 42)

    my_model = KNN(n_neighbors = 5)
    sklearn_model = KNeighborsClassifier(n_neighbors = 5)

    my_model.fit(X_train, y_train)
    sklearn_model.fit(X_train, y_train)

    my_predictions = my_model.predict(X_test)
    sklearn_predictions = sklearn_model.predict(X_test)

    # Признавам си, моят модел беше твърде бавен, за да изчакам predictions, при положение че при score ще ги изчисли наново
    print("My predictions: ", my_predictions)
    print("Sklearn predictions: ", sklearn_predictions)

    my_score = my_model.score(X_test, y_test)
    sklearn_score = sklearn_model.score(X_test, y_test)

    # Но пък резултатите им са напълно еднакви
    print("My score: ", my_score)
    print("Sklearn score: ", sklearn_score)

    # Като цяло, ако изключим факта, че моят модел е безумно бавен, върши точно същата работа

    
if __name__ == "__main__":
    main()