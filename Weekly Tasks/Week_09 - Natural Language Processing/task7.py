from nltk import tokenize, chunk, tag


def main():
    article = ""
    with open("../DATA/article_uber.txt", "r") as f:
        article = f.read()

    article_sents = tokenize.sent_tokenize(article)
    words_per_sent = [tokenize.word_tokenize(sent) for sent in article_sents]
    pos_tagged_words_per_sent = tag.pos_tag_sents(words_per_sent)
    print(pos_tagged_words_per_sent[-1])

    ner_sents = list(chunk.ne_chunk_sents(pos_tagged_words_per_sent, binary=True))
    print("First sentence with NER applied:", ner_sents[0])

    print("All chunks with label NE:")
    for ner_sent in ner_sents:
        for token in ner_sent:
            if hasattr(token, "label") and token.label() == "NE":
                print(f"\t{token}")


if __name__ == "__main__":
    main()