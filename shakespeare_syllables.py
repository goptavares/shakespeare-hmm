#!/usr/bin/python

"""
shakespeare_syllables.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import string

import hmm_end_state
import util

from hyphen import Hyphenator


def main():
    # Load Shakespeare dataset.
    sonnets = util.loadShakespeareSonnets()
    tokens = util.getUniqueSyllables(sonnets)
    numObs = len(tokens)
    numStates = 4
    model = hmm_end_state.HMM(numStates, numObs)

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
    model.train(words, maxIter=4)

    # Generate artificial sonnet with any generated words and detokenize it.
    artificialSonnet = model.generateSonnetFromSyllables(numSentences=14,
                                                         numWordsPerSentence=8)
    detokenizedSonnet = []
    for sentence in artificialSonnet:
        detokenizedSentence = []
        for w, word in enumerate(sentence):
            detokenizedWord = ''
            if w == 0:
                syll = word[0]
                detokenizedWord += tokens[syll][0].upper() + tokens[syll][1:]
                for syll in word[1:]:
                    detokenizedWord += tokens[syll]
            else:
                for syll in word:
                    detokenizedWord += tokens[syll]
            detokenizedSentence.append(detokenizedWord)
        detokenizedSonnet.append(detokenizedSentence)

    # Write detokenized sonnet to text file.
    util.writeSonnetToTxt(detokenizedSonnet)

    # Generate artificial sonnet with only valid words and detokenize it.
    artificialSonnet = model.generateSonnetFromSyllables(
        numSentences=14, numWordsPerSentence=8,
        validWords=util.getUniqueWords(sonnets), tokens=tokens)
    detokenizedSonnet = []
    for sentence in artificialSonnet:
        detokenizedSentence = []
        for w, word in enumerate(sentence):
            detokenizedWord = ''
            if w == 0:
                syll = word[0]
                detokenizedWord += tokens[syll][0].upper() + tokens[syll][1:]
                for syll in word[1:]:
                    detokenizedWord += tokens[syll]
            else:
                for syll in word:
                    detokenizedWord += tokens[syll]
            detokenizedSentence.append(detokenizedWord)
        detokenizedSonnet.append(detokenizedSentence)

    # Write detokenized sonnet to text file.
    util.writeSonnetToTxt(detokenizedSonnet)


if __name__ == '__main__':
    main()