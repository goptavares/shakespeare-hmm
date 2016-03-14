#!/usr/bin/python

"""
brazbanian.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import string

import hmm
import util


def main():
    # Load Moraes dataset.
    poemsMoraes = util.loadMoraesPoems()
    tokensMoraes = util.getUniqueWords(poemsMoraes)
    numObs = len(tokensMoraes)
    numStates = 50
    modelMoraes = hmm.HMM(numStates, numObs)

    # Train model on tokenized dataset.
    sentences = []
    for poem in poemsMoraes:
        for sentence in poem:
            tokenizedSentence = []
            for word in sentence:
                tokenizedSentence.append(tokensMoraes.index(word.lower())) 
            sentences.append(tokenizedSentence)
    modelMoraes.train(sentences, maxIter=100)

    # Load Kadare dataset.
    poemsKadare = util.loadKadarePoems()
    tokensKadare = util.getUniqueWords(poemsKadare)
    numObs = len(tokensKadare)
    numStates = 50
    modelKadare = hmm.HMM(numStates, numObs)

    # Train model on tokenized dataset.
    sentences = []
    for poem in poemsKadare:
        for sentence in poem:
            tokenizedSentence = []
            for word in sentence:
                tokenizedSentence.append(tokensKadare.index(word.lower())) 
            sentences.append(tokenizedSentence)
    modelKadare.train(sentences, maxIter=100)

    # Generate artificial poem and detokenize it.
    detokenizedPoem = []
    for s in xrange(12):
        if not s % 2:
            sentence = modelMoraes.generateSonnetFromWords(
                numSentences=1, numWordsPerSentence=8)[0]
            detokenizedSentence = []
            for i, word in zip(xrange(len(sentence)), sentence):
                if i == 0:
                    detokenizedSentence.append(
                        string.capwords(tokensMoraes[word]))
                else:
                    detokenizedSentence.append(tokensMoraes[word])
            detokenizedPoem.append(detokenizedSentence)
        else:
            sentence = modelKadare.generateSonnetFromWords(
                numSentences=1, numWordsPerSentence=8)[0]
            detokenizedSentence = []
            for i, word in zip(xrange(len(sentence)), sentence):
                if i == 0:
                    detokenizedSentence.append(
                        string.capwords(tokensKadare[word]))
                else:
                    detokenizedSentence.append(tokensKadare[word])
            detokenizedPoem.append(detokenizedSentence)

    # Write detokenized poem to text file.s
    util.writeSonnetToTxt(detokenizedPoem)


if __name__ == '__main__':
    main()