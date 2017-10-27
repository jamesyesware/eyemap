import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import numpy as np
import time
from s3_parser import EmailParser

class Classifier:
    stemmer = LancasterStemmer()
    parser = EmailParser()
    ignore_words = ['?']

    def __init__(self, samples_count=400):
        self.samples_count = samples_count
        self.training_data = self.parser.fetch_data('data.json', self.samples_count)

        self.synapse_0 = None
        self.synapse_1 = None

        self.words = []
        self.classes = []
        self.documents = []
        self.training = []
        self.output = []


    def prepare_bag_of_words(self):
        # loop through each sentence in our training data
        for pattern in self.training_data:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern['subject'])
            # add to our words list
            self.words.extend(w)
            # add to documents in our corpus
            self.documents.append((w, pattern['auto_reply']))
            # add to our classes list
            if pattern['auto_reply'] not in self.classes:
                self.classes.append(pattern['auto_reply'])

        # stem and lower each word and remove duplicates
        self.words = [self.stemmer.stem(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = list(set(self.words))

        # remove duplicates
        self.classes = list(set(self.classes))

        # create an empty array for our output
        output_empty = [0] * len(self.classes)

        # training set, bag of words for each sentence
        for doc in self.documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [self.stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            self.training.append(bag)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            self.output.append(output_row)


    # compute sigmoid nonlinearity
    def sigmoid(self, x):
        output = 1/(1+np.exp(-x))
        return output


    # convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(self, output):
        return output*(1-output)


    def clean_up_sentence(self, sentence):
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words


    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow_onehot(self, sentence):
        # tokenize the patter
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0]*len(self.words)
        for s in sentence_words:
            for i,w in enumerate(self.words):
                if w == s:
                    bag[i] = 1

        return(np.array(bag))


    def think(self, sentence):
        x = self.bow_onehot(sentence.lower())
        # input layer is our bag of words
        l0 = x
        # matrix multiplication of input and hidden layer
        l1 = self.sigmoid(np.dot(l0, self.synapse_0))
        # output layer
        l2 = self.sigmoid(np.dot(l1, self.synapse_1))
        return l2


    def train(self, X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
        print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
        print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(self.classes)) )
        np.random.seed(1)

        last_mean_error = 1
        # randomly initialize our weights with mean 0
        synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
        synapse_1 = 2*np.random.random((hidden_neurons, len(self.classes))) - 1

        prev_synapse_0_weight_update = np.zeros_like(synapse_0)
        prev_synapse_1_weight_update = np.zeros_like(synapse_1)

        synapse_0_direction_count = np.zeros_like(synapse_0)
        synapse_1_direction_count = np.zeros_like(synapse_1)

        for j in iter(range(epochs+1)):
            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = self.sigmoid(np.dot(layer_0, synapse_0))

            if(dropout):
                layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

            layer_2 = self.sigmoid(np.dot(layer_1, synapse_1))

            # how much did we miss the target value?
            layer_2_error = y - layer_2

            if (j% 10000) == 0 and j > 5000:
                # if this 10k iteration's error is greater than the last iteration, break out
                if np.mean(np.abs(layer_2_error)) < last_mean_error:
                    print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                    last_mean_error = np.mean(np.abs(layer_2_error))
                else:
                    print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                    break

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_2_delta = layer_2_error * self.sigmoid_output_to_derivative(layer_2)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            layer_1_error = layer_2_delta.dot(synapse_1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * self.sigmoid_output_to_derivative(layer_1)

            synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
            synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

            if(j > 0):
                synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
                synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))

            synapse_1 += alpha * synapse_1_weight_update
            synapse_0 += alpha * synapse_0_weight_update

            prev_synapse_0_weight_update = synapse_0_weight_update
            prev_synapse_1_weight_update = synapse_1_weight_update

        now = datetime.datetime.now()

        # persist synapses
        synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
                   'datetime': now.strftime("%Y-%m-%d %H:%M"),
                   'words': self.words,
                   'classes': self.classes
                  }

        synapse_file = "synapses.json"

        with open(synapse_file, 'w') as outfile:
            json.dump(synapse, outfile, indent=4, sort_keys=True)

        print ("saved synapses to:", synapse_file)


    def start_training(self, epochs=10000):
        self.prepare_bag_of_words()
        X = np.array(self.training)
        y = np.array(self.output)

        start_time = time.time()

        self.train(X, y, hidden_neurons=20, alpha=0.1, epochs=epochs, dropout=False, dropout_percent=0.2)

        elapsed_time = time.time() - start_time
        print ("Training Complete in ", elapsed_time, " seconds")


    def hydrate_synapses(self):
        # load our calculated synapse values
        synapse_file = 'synapses.json'
        with open(synapse_file) as data_file:
            synapses = json.load(data_file)
            self.synapse_0 = np.asarray(synapses['synapse0'])
            self.synapse_1 = np.asarray(synapses['synapse1'])

    def classify(self, sentence):
        # probability threshold
        ERROR_THRESHOLD = 0.2

        if not np.any(self.synapse_0) or not np.any(self.synapse_1):
            self.hydrate_synapses()

        results = self.think(sentence)

        results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
        results.sort(key=lambda x: x[1], reverse=True)
        return_results =[[self.classes[r[0]],r[1]] for r in results]
        print ("%s \n classification: %s" % (sentence, return_results))
        return return_results

    def practice_run(self):
        self.classify("Ann, people found you in search this week")
        self.classify("Do you like horror movies?")
        self.classify("RE: Snowflake/Microsoft - potential use case w\Wipro")

        self.classify("Delivery Status Notification (Failure?")
        self.classify("Automatic reply: Deliveroo for Business Reviews and Opportunities")
        self.classify("Accepted: Whistle Sports | Simply Measured @ Wed Oct 25, 2017 8am - 9am (tazi.flory@simplymeasured.com)")

