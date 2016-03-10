#!/usr/bin/python

"""
shakespeare_syllables.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import string

import hmm
import util

from hyphen import Hyphenator


def main():
    # Load Shakespeare dataset.
    sonnets = util.loadShakespeareSonnets()
    tokens = util.getUniqueSyllables(sonnets)
    numObs = len(tokens)
    numStates = 20
    model = hmm.HMM(numStates, numObs)

    # Train model on tokenized dataset.
    h = Hyphenator('en_GB')
    words = []
    for sonnet in sonnets:
        for sentence in sonnet:
            for word in sentence:
                tokenizedWord = []
                syllables = h.syllables(unicode(word.lower()))
                if len(syllables) < 2:
                    tokenizedWord.append(tokens.index(unicode(word.lower()))) 
                else:
                    for syllable in syllables:
                        tokenizedWord.append(tokens.index(syllable))
            words.append(tokenizedWord)
    model.train(words)


if __name__ == '__main__':
    main()