#!/usr/bin/python

"""
shakespeare_sentences.py
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
    tokens = util.getUniqueSentences(sonnets)
    numObs = len(tokens)
    numStates = 50
    model = hmm.HMM(numStates, numObs)

    # Train model on tokenized dataset.
    tokenizedSonnets = []
    for sonnet in sonnets:
        tokenizedSonnet = []
        for sentence in sonnet:
            tokenizedSonnet.append(tokens.index(sentence))
        tokenizedSonnets.append(tokenizedSonnet)
    model.train(tokenizedSonnets, maxIter=50)

    # Generate artificial sonnet and detokenize it.
    artificialSonnet = model.generateSonnetFromSentences(numSentences=14)
    detokenizedSonnet = []
    for sentence in artificialSonnet:
        detokenizedSonnet.append(tokens[sentence])

    # Write detokenized sonnet to text file.
    util.writeSonnetToTxt(detokenizedSonnet)


if __name__ == '__main__':
    main()