from os import pipe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sympy import Nor


def main():
    data = []
    with open("../DATA/price_movements.txt", "r") as file:
        data = file.read()
        data = np.array(ast.literal_eval(data))

    print(data.shape)

    companies = ['Apple', 'AIG', 'Amazon', 'American express', 'Boeing', 'Bank of America', 'British American Tobacco', 'Canon', 'Caterpillar', 'Colgate-Palmolive', 'ConocoPhillips', 'Cisco', 'Chevron', 'DuPont de Nemours', 'Dell', 'Ford', 'General Electrics', 'Google/Alphabet', 'Goldman Sachs', 'GlaxoSmithKline', 'Home Depot', 'Honda', 'HP', 'IBM', 'Intel', 'Johnson & Johnson', 'JPMorgan Chase', 'Kimberly-Clark', 'Coca Cola', 'Lookheed Martin', 'MasterCard', 'McDonalds', '3M', 'Microsoft', 'Mitsubishi', 'Navistar', 'Northrop Grumman', 'Novartis', 'Pepsi', 'Pfizer', 'Procter Gamble', 'Philip Morris', 'Royal Dutch Shell', 'SAP', 'Schlumberger', 'Sony', 'Sanofi-Aventis', 'Symantec', 'Toyota', 'Total', 'Taiwan Semiconductor Manufacturing', 'Texas instruments', 'Unilever', 'Valero Energy', 'Walgreen', 'Wells Fargo', 'Wal-Mart', 'Exxon', 'Xerox', 'Yahoo']

    pipeline = Pipeline(steps=[
        ("normalizer", Normalizer()),
        ("kmeans", KMeans(n_clusters=10))
    ])
    pipeline.fit(data)

    labels_companies_df = pd.DataFrame({"labels":pipeline["kmeans"].labels_, "companies":companies})
    labels_companies_df = labels_companies_df.sort_values(by="labels")
    print(labels_companies_df)


if __name__ == "__main__":
    main()