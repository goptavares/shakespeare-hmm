#!/usr/bin/python

"""
shakespeare_cross_val.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import string

import hmm
import util


def main():
    # Load Shakespeare dataset.
    sonnets = util.loadShakespeareSonnets()

    # Split dataset into training and testing.
    sonnetsTrain = sonnets[:120]
    sonnetsTest = sonnets[120:]

    tokens = util.getUniqueWords(sonnets)
    numObs = len(tokens)

    numStates = [4, 6, 8, 10]
    logLikelihoods = []
    for n in numStates:
        model = hmm.HMM(n, numObs)

        # Train model on tokenized training dataset.
        sentences = []
        for sonnet in sonnetsTrain:
            for sentence in sonnet:
                tokenizedSentence = []
                for word in sentence:
                    tokenizedSentence.append(tokens.index(word.lower())) 
                sentences.append(tokenizedSentence)
        model.train(sentences, maxIter=80)

        # Calculate log-likelihood of validation dataset for the model.
        logLikelihood = 0
        for sonnet in sonnetsTest:
            for sentence in sonnet:
                tokenizedSentence = []
                for word in sentence:
                    tokenizedSentence.append(tokens.index(word.lower()))
                alphas = model.forward(tokenizedSentence)
                if np.sum(alphas[:,-1]) != 0:
                    logLikelihood += np.log(np.sum(alphas[:,-1]))
        logLikelihoods.append(logLikelihood)
        print("States: " + str(n) + ", LL: " + str(logLikelihood))

    # Get optimal number of states and create best model.
    print("Optimal number of states: " +
          str(numStates[np.argmax(logLikelihoods)]))


if __name__ == '__main__':
    main()