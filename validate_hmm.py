#!/usr/bin/python

"""
validate_hmm.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import string

import hmm
import util


def validateModelWithArtificialDataset():
    sequences = []
    with open('./Example_1.txt', 'r') as f:
        linesSkipped = 2
        for line in f:
            if linesSkipped == 2:
                linesSkipped = 0
                sequences.append([int(n)-1 for n in line.strip().split(' ')])
                continue
            else:
                linesSkipped += 1
    f.close()

    tokens = set()
    for seq in sequences:
        tokens |= set(seq)
    numObs = len(tokens)
    numStates = 2
    model = hmm.HMM(numStates, numObs)
    model.train(sequences, maxIter=20, threshold=0.0001)

    print model.A
    print model.B
    print model.I


def validateModelWithShakespeareDataset():
    # Load Shakespeare dataset.
    sonnets = util.loadShakespeareSonnets()

    # Split dataset into training and testing.
    sonnetsTrain = sonnets[:120]
    sonnetsTest = sonnets[120:]

    tokens = util.getUniqueWords(sonnets)
    numObs = len(tokens)

    numStates = 8
    model = hmm.HMM(numStates, numObs)

    # Train model on tokenized training dataset.
    sentences = []
    for sonnet in sonnetsTrain:
        for sentence in sonnet:
            tokenizedSentence = []
            for word in sentence:
                tokenizedSentence.append(tokens.index(word.lower())) 
            sentences.append(tokenizedSentence)

    for i in xrange(30):
        if i == 0:
            model.train(sentences, maxIter=1, randomInit=True)
        else:
            model.train(sentences, maxIter=1, randomInit=False)

        # Calculate log-likelihood of training dataset for the current model.
        NLL = 0
        for sonnet in sonnetsTrain:
            for sentence in sonnet:
                tokenizedSentence = []
                for word in sentence:
                    tokenizedSentence.append(tokens.index(word.lower()))
                alphas = model.forward(tokenizedSentence)
                if np.sum(alphas[:,-1]) != 0:
                    NLL -= np.log(np.sum(alphas[:,-1]))
        print("Iteration: " + str(i))
        print("NLL training: " + str(NLL))

        # Calculate log-likelihood of validation dataset for the current model.
        NLL = 0
        for sonnet in sonnetsTest:
            for sentence in sonnet:
                tokenizedSentence = []
                for word in sentence:
                    tokenizedSentence.append(tokens.index(word.lower()))
                alphas = model.forward(tokenizedSentence)
                if np.sum(alphas[:,-1]) != 0:
                    NLL -= np.log(np.sum(alphas[:,-1]))
        print("NLL test: " + str(NLL))
