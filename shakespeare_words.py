#!/usr/bin/python

"""
shakespeare_words.py
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
    tokens = util.getUniqueWords(sonnets)
    numObs = len(tokens)
    numStates = 6
    model = hmm.HMM(numStates, numObs)

    # Train model on tokenized dataset.
    sentences = []
    for sonnet in sonnets:
        for sentence in sonnet:
            tokenizedSentence = []
            for word in sentence:
                tokenizedSentence.append(tokens.index(word.lower())) 
            sentences.append(tokenizedSentence)
    model.train(sentences, maxIter=10)

    # Generate artificial sonnet and detokenize it.
    artificialSonnet = model.generateSonnetFromWords(numSentences=14,
                                                     numWordsPerSentence=8)
    detokenizedSonnet = []
    for sentence in artificialSonnet:
        detokenizedSentence = []
        for i, word in zip(xrange(len(sentence)), sentence):
            if i == 0:
                detokenizedSentence.append(string.capwords(tokens[word]))
            else:
                detokenizedSentence.append(tokens[word])
        detokenizedSonnet.append(detokenizedSentence)

    # Write detokenized sonnet to text file.
    util.writeSonnetToTxt(detokenizedSonnet)


if __name__ == '__main__':
    main()