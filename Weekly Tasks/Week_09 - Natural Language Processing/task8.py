from nltk import tokenize, tag, chunk
import matplotlib.pyplot as plt
import collections


def main():
    articles = []
    with open("../DATA/article_uber.txt", "r") as f:
        article = f.read()

    sents_tokenized = tokenize.sent_tokenize(article)
    words_per_sent = [tokenize.word_tokenize(sent) for sent in sents_tokenized]
    pos_tagged_words_per_sent = tag.pos_tag_sents(words_per_sent)

    ner_categories = collections.defaultdict(int)
    for sent in chunk.ne_chunk_sents(pos_tagged_words_per_sent):
        for current_chunk in sent:
            if hasattr(current_chunk, "label"):
                ner_categories[current_chunk.label()] += 1

    print(ner_categories)

    plt.title('Distribution of NER categories')
    plt.pie(ner_categories.values(), labels=ner_categories, autopct='%1.1f%%', startangle=140)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
