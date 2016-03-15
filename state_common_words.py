#!/usr/bin/python

"""
state_common_words.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import string
import nltk.tag.hmm as nth

import hmm
import util


def main():
    # Load Shakespeare dataset.
    sonnets = util.loadShakespeareSonnets()
    tokens = util.getUniqueWords(sonnets)
    numObs = len(tokens)
    numStates = 8

    # Get dataset.
    sentences = []
    for sonnet in sonnets:
        for sentence in sonnet:
            tokenizedSentence = []
            for word in sentence:
                tokenizedSentence.append((word.lower(),''))
            sentences.append(tokenizedSentence)

    trainer = nth.HiddenMarkovModelTrainer(range(numStates),tokens)
    model = trainer.train_unsupervised(sentences, max_iterations=20)

    freq = {}
    for token in tokens:
        freq[token] = 0
    for sonnet in sonnets:
        for sentence in sonnet:
            for word in sentence:
                freq[word.lower()] += 1

    for i in xrange(numStates):
        print "State: " + str(i)
        s = np.argsort(model._outputs[i]._data)
        s = np.flipud(s)
        l = []
        for j in xrange(5):
            d = model._outputs[i]._sample_dict
            l.append(d.keys()[d.values().index(s[j])])
        print ','.join(l)


if __name__ == '__main__':
    main()