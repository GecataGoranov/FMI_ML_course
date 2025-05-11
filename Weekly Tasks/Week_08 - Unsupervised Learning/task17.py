import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    docs = ['cats say meow', 'dogs say woof', 'dogs chase cats']

    tf_idf = TfidfVectorizer()
    tf_idf.fit(docs)
    transformed = tf_idf.transform(docs)

    print(transformed.toarray())
    print(tf_idf.get_feature_names_out())


if __name__ == "__main__":
    main()