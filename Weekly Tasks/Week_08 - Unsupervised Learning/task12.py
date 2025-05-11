import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

from sklearn.preprocessing import normalize
from sklearn import manifold


def main():
    data = []
    with open("../DATA/price_movements.txt", "r") as file:
        data = file.read()
        data = np.array(ast.literal_eval(data))

    companies = ['Apple', 'AIG', 'Amazon', 'American express', 'Boeing', 'Bank of America', 'British American Tobacco', 'Canon', 'Caterpillar', 'Colgate-Palmolive', 'ConocoPhillips', 'Cisco', 'Chevron', 'DuPont de Nemours', 'Dell', 'Ford', 'General Electrics', 'Google/Alphabet', 'Goldman Sachs', 'GlaxoSmithKline', 'Home Depot', 'Honda', 'HP', 'IBM', 'Intel', 'Johnson & Johnson', 'JPMorgan Chase', 'Kimberly-Clark', 'Coca Cola', 'Lookheed Martin', 'MasterCard', 'McDonalds', '3M', 'Microsoft', 'Mitsubishi', 'Navistar', 'Northrop Grumman', 'Novartis', 'Pepsi', 'Pfizer', 'Procter Gamble', 'Philip Morris', 'Royal Dutch Shell', 'SAP', 'Schlumberger', 'Sony', 'Sanofi-Aventis', 'Symantec', 'Toyota', 'Total', 'Taiwan Semiconductor Manufacturing', 'Texas instruments', 'Unilever', 'Valero Energy', 'Walgreen', 'Wells Fargo', 'Wal-Mart', 'Exxon', 'Xerox', 'Yahoo']

    data_normalized = normalize(data, axis=1)

    learning_rates = [20, 50, 100, 200, 1000]
    
    for rate in learning_rates:
        tsne = manifold.TSNE(learning_rate=100)
        transformed = tsne.fit_transform(data)

        plt.scatter(transformed[:,0], transformed[:,1], alpha=0.4)

        i=0
        for el in transformed:
            plt.annotate(companies[i], (el[0], el[1]))
            i += 1

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()