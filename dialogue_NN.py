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
            for x in range(size):
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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_output_to_derivative(output):
    return output * (1-output)


"""* * * TRAINING * * *"""
def init_synapses(X, hidden_neurons, classes):
    """Initializes our synapses (using random values)."""
    # Ensures we have a "consistent" randomness for convenience.
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    return synapse_0, synapse_1


def feedforward(X, synapse_0, synapse_1):
    """Feed forward through layers 0, 1, and 2."""
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    return layer_0, layer_1, layer_2


def get_synapses(epochs, X, y, alpha, synapse_0, synapse_1):
    """Update our weights for each epoch."""
    # Initializations.
    last_mean_error = 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    synapse_0_direction_count = np.zeros_like(synapse_0)

    prev_synapse_1_weight_update = np.zeros_like(synapse_1)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    # Make an iterator out of the number of epochs we requested.
    for j in iter(range(epochs+1)):
        layer_0, layer_1, layer_2 = feedforward(X, synapse_0, synapse_1)

        # How much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # If this 10k iteration's error is greater than the last iteration,
            # break out.
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break

        # In what direction is the target value?  How much is the change for layer_2?
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # How much did each l1 value contribute to the l2 error (according to the weights)?
        # (Note: .T means transpose and can be accessed via numpy!)
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # In what direction is the target l1?  How much is the change for layer_1?
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        # Manage updates.
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if j > 0:
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    return synapse_0, synapse_1


def save_synapses(filename, words, classes, synapse_0, synapse_1):
    """Save our weights as a JSON file for later use."""
    now = datetime.datetime.now()

    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print("Saved synapses to:", synapse_file)


def train(X, y, words, classes, hidden_neurons=10, alpha=1, epochs=50000):
    """Train using specified parameters."""
    print("Training with {0} neurons and alpha = {1}".format(hidden_neurons, alpha))

    synapse_0, synapse_1 = init_synapses(X, hidden_neurons, classes)

    # For each epoch, update our weights
    synapse_0, synapse_1 = get_synapses(epochs, X, y, alpha, synapse_0, synapse_1)

    # Save our work
    save_synapses("synapses.json", words, classes, synapse_0, synapse_1)


def start_training(words, classes, training_data, output):
    """Initialize training process and keep track of processing time."""
    start_time = time.time()
    X = np.array(training_data)
    y = np.array(output)

    train(X, y, words, classes, hidden_neurons=20, alpha=0.1, epochs=100000)

    elapsed_time = time.time() - start_time
    print("Processing time:", elapsed_time, "seconds")


def main():

    stemmer = LancasterStemmer()
    raw_training_data = get_raw_training_data('dialogue_data.csv')
    words, documents, classes = organize_raw_training_data(raw_training_data, stemmer)
    print(words)
    print(documents)
    print(classes)

    #TODO: WE NEED TO FIGURE OUT WHAT 'OUTPUT' IS FOR START_TRAINING
    #TODO: WE NEED TO CLASSIFY NEW SENTENCES




if __name__ == "__main__":
    main()
