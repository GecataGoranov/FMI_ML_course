import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def main():
    tweets = ['This is the best #nlp exercise ive found online! #python', '#NLP is super fun! <3 #learning', 'Thanks @datacamp :) #nlp #python']

    print("All hashtags in first tweet:", re.findall(r"#\S+", tweets[0]))
    print("All mentions and hashtags in last tweet:", re.findall(r"#\S+|@\S+", tweets[-1]))
    
    tokens = []
    for tweet in tweets:
        tokens.append(re.findall(r"\w+|[,:'!;]", tweet))

    print("All tokens:", tokens)


if __name__ == "__main__":
    main()