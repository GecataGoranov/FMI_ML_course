from ast import Constant
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():
    news = pd.read_csv("../DATA/fake_or_real_news.csv", index_col=0)

    X = news.drop(columns=["label"])
    y = news["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=52)

    bow = CountVectorizer(stop_words="english")
    bow_news_train = bow.fit_transform(X_train["text"])
    bow_news_test = bow.transform(X_test["text"])

    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    tfidf_news_train = tfidf_vectorizer.fit_transform(X_train["text"])
    tfidf_news_test = tfidf_vectorizer.transform(X_test["text"])

    bag_of_words_nb = MultinomialNB()
    tfidf_nb = MultinomialNB()

    bag_of_words_nb.fit(bow_news_train, y_train)
    tfidf_nb.fit(tfidf_news_train, y_train)

    print("Accuracy when using BoW:", bag_of_words_nb.score(bow_news_test, y_test))
    print("Accuracy when using TF-IDF:", tfidf_nb.score(tfidf_news_test, y_test))

    param_grid = {
        "alpha" : np.linspace(0, 1, 11)
    }

    grid_search = GridSearchCV(
        estimator=MultinomialNB(),
        param_grid=param_grid,
    )

    grid_search.fit(tfidf_news_train, y_train)

    # This doesn't match for some reason, although the latter does
    fake_probs_df = pd.DataFrame({"feature" : tfidf_vectorizer.get_feature_names_out(), "prob" : grid_search.best_estimator_.feature_log_prob_[1]})
    print("FAKE", list(fake_probs_df.sort_values(by="prob", ascending=True).head(20)["feature"]))

    real_probs_df = pd.DataFrame({"feature" : tfidf_vectorizer.get_feature_names_out(), "prob" : grid_search.best_estimator_.feature_log_prob_[0]})
    print("REAL", list(real_probs_df.sort_values(by="prob", ascending=False).head(20)["feature"]))

    y_pred_bow = grid_search.predict(bow_news_test)
    y_pred_tfidf = grid_search.predict(tfidf_news_test)

    bow_cm = confusion_matrix(y_test, y_pred_bow)
    tfidf_cm = confusion_matrix(y_test, y_pred_tfidf)

    fig, axes = plt.subplots(1, 2, sharey = "row")

    disp1 = ConfusionMatrixDisplay(bow_cm)
    disp1.plot(ax = axes[0])

    disp2 = ConfusionMatrixDisplay(tfidf_cm)
    disp2.plot(ax = axes[1])

    fig.suptitle("Confusion matrix: BoW (left) vs TF-IDF (right)")

    plt.show()

    plt.plot(param_grid["alpha"], grid_search.cv_results_["mean_test_score"])
    plt.xticks(param_grid["alpha"])
    plt.grid(visible=True)
    plt.show()

    """
    Well, the results differ from yours...
    """


if __name__ == "__main__":
    main()