#!/usr/bin/python

"""
shakespeare.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np

import hmm
import util

def main():
    # Load Shakespeare data set.
    sonnets = util.loadShakespeareSonnets()
    tokens = util.getUniqueWords(sonnets)
    numObs = len(tokens)
    numStates = 10
    model = hmm.HMM(numStates, numObs)

    sentences = []
    for sonnet in sonnets:
        for sentence in sonnet:
            tokenizedSentece = []
            for word in sentence:
                tokenizedSentece.append(tokens.index(word)) 
            sentences.append(tokenizedSentece)
    # print(len(sentences))
    # print(sentences[0])
    # print(sentences[100])
    model.train(sentences)


if __name__ == '__main__':
    main()