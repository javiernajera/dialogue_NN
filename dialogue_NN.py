import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import csv
import numpy as np
import time
import re


def create_training_data(stems, classes, documents):
    training_data = []
    output = []
    size = len(classes)
    for document in documents:
        bag = []
        target = []
        for word in document[0]:
            if word in stems:
                bag.append(1)
            else:
                bag.append(0)
        if document[0] in classes:
            index = classes.index(document[0])
            for x in range(size)
                if x == index:
                    target.append(1)
                else:
                    target.append(0)
        else:
            for x in range(size):
                target.append(0)


        training_data.append(bag)
        output.append(target)
return training_data, output


def preprocess_words(words, stemmer):
    stem_list = []
    for word in words:
        if word not in stem_list and word is not "?":
            stem_list.append(stemmer.stem(word))

    return stem_list

def get_raw_training_data(fileName):
    raw_training_data = []

    pattern = re.compile('"(.*)","(.*)"')
    with open(fileName, newline='') as csvfile:
        dialogue = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in dialogue:
            text = ' '.join(row)
            match = pattern.match(text)
            raw_training_data.append({"person": match.group(1), "sentence": match.group(2)})

            #raw_training_data.append({"person" : text_arr[0], "sentence" : text_arr[1]})
    return raw_training_data

def organize_raw_training_data(training_data, stemmer):
    words = []
    documents = []
    classes = []
    for element in training_data:
        tokens = nltk.word_tokenize(element['sentence'])
        person = element['person']
        words.extend(tokens)
        documents.append((tokens, person))
        if person not in classes:
            classes.append(person)

    words = preprocess_words(words, stemmer)
    return words, documents, classes



def main():
    # nltk.download('punkt')
    stemmer = LancasterStemmer()
    raw_training_data = get_raw_training_data('dialogue_data.csv')
    words, documents, classes = organize_raw_training_data(raw_training_data, stemmer)
    print(words)
    print(documents)
    print(classes)


if __name__ == "__main__":
    main()
