import spacy

from nltk import tokenize, tag, chunk

def main():
    article = ""
    with open("../DATA/article_uber.txt", "r") as f:
        article = f.read()

    nlp_pipeline = spacy.load('en_core_web_sm')
    nlp_pipeline.get_pipe("ner")

    doc = nlp_pipeline(article)
    for i in range(len(doc.ents)):
        print(f"{doc.ents[i].label_} {doc.ents[i]}")

    # sents_tokenized = tokenize.sent_tokenize(article)
    # words_per_sent = [tokenize.word_tokenize(sent) for sent in sents_tokenized]
    # pos_tagged_words_per_sent = tag.pos_tag_sents(words_per_sent)

    """
    I can't without tasks 7 and 8.
    """


if __name__ == "__main__":
    main()