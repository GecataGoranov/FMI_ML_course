import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def main():
    article = ""
    with open("../DATA/article.txt", "r") as f:
        article = f.read()

    counter = collections.Counter(word_tokenize(article))
    print(counter.most_common(10))

    """
    That way there are many words, that don't hold much information so we can't say anything about the topics
    """

    stop_words = set(stopwords.words("english"))
    raw_tokens = word_tokenize(article)
    no_stopword_tokens = [w for w in raw_tokens if w.lower() not in stop_words]
    no_stopword_alphabetic_tokens = [w for w in no_stopword_tokens if re.match(r"\w+", w)]

    lemmatizer = WordNetLemmatizer() 
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in no_stopword_alphabetic_tokens]

    counter = collections.Counter(lemmatized_tokens)
    print(counter.most_common(10))

    """
    Now we can see, that the article has something to do with programming.
    """


if __name__ == "__main__":
    main()