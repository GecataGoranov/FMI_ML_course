import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def main():
    news = pd.read_csv("../DATA/fake_or_real_news.csv", index_col=0)
    print(news.head(7))
    distribution = pd.concat([news["label"].value_counts(), news["label"].value_counts(normalize=True)], axis=1)
    print("Distribution of labels", distribution)

    X = news.drop(columns=["label"])
    y = news["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=52)

    count_vectorizer = CountVectorizer(stop_words="english")
    count_vectorized_news = count_vectorizer.fit_transform(X_train["text"])

    print("First 10 tokens:", count_vectorizer.get_feature_names_out()[:10])
    print("Size of vocabulary:", len(count_vectorizer.vocabulary_))

    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    tfidf_news = tfidf_vectorizer.fit_transform(X_train["text"])
    
    print(tfidf_vectorizer.get_feature_names_out()[:10])
    print(tfidf_news.toarray()[:5])

    df_count = pd.DataFrame(count_vectorized_news.toarray(), columns=count_vectorizer.get_feature_names_out())
    df_tfidf = pd.DataFrame(tfidf_news.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    print("DataFrame obtained by CountVectorizer - df_count:", df_count.head(5))
    print("DataFrame obtained by TfidfVectorizer - df_tfidf:", df_tfidf.head(5))

    print("Tokens that are in df_count, but are not in df_tfidf:", set(count_vectorizer.get_feature_names_out()) - set(tfidf_vectorizer.get_feature_names_out()))

if __name__ == "__main__":
    main()