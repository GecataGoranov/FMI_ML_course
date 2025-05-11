import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel


def main():
    articles = []
    with open("../DATA/messy_articles.txt", "r") as f:
        data = f.read()
        articles = ast.literal_eval(data)

    dictionary = Dictionary(articles)
    corpus = [dictionary.doc2bow(article) for article in articles]

    print('Id of "computer":', dictionary.token2id["computer"])
    print("First 10 ids and frequency counts from the 5th document:", dictionary.doc2bow(articles[4])[:10])
    print("Top 5 words in the 5th document:", [dictionary[x[0]] for x in sorted(dictionary.doc2bow(articles[4]), key=lambda tup: tup[1], reverse=True)[:5]])
    print("Top 5 words across all documents:", [(dictionary[x[0]], x[1]) for x in sorted(dictionary.doc2bow(flatten(articles)), key=lambda tup: tup[1], reverse=True)[:5]])

    tfidf = TfidfModel(corpus)
    print("First 5 term ids with their weights:", tfidf[corpus[4]][:5])
    print("Top 5 words in the 5th document when using tf-idf:", [dictionary[x[0]] for x in sorted(tfidf[corpus[4]], key=lambda tup: tup[1], reverse=True)[:5]])  


def flatten(xss):
    return [x for xs in xss for x in xs]


if __name__ == "__main__":
    main()