#!/usr/bin/python

"""
util.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import codecs
import datetime
import os
import re

from hyphen import Hyphenator


def loadShakespeareSonnets():
    sonnets = []
    with open('./shakespeare.txt', 'r') as f:
        sonnet = []
        sonnetToAppend = False
        for line in f:
            if line.strip().split(' ')[-1].isdigit():
                sonnetToAppend = True
                continue
            if line == '\n':
                if sonnetToAppend:
                    sonnets.append(sonnet)
                    sonnet = []
                    sonnetToAppend = False
                continue
            sonnet.append([re.sub(r'[^\w\s\']', '', w) for
                           w in line.strip().split(' ')])
        sonnets.append(sonnet)
    f.close()
    return sonnets


def loadSpenserSonnets():
    sonnets = []
    with open('./spenser.txt', 'r') as f:
        sonnet = []
        sonnetToAppend = False
        skipLine = False
        for line in f:
            if line != '\n' and len(line.strip().split(' ')) == 1:
                skipLine = True
                sonnetToAppend = True
                continue
            if line == '\n':
                if skipLine:
                    skipLine = False
                    continue
                if sonnetToAppend:
                    sonnets.append(sonnet)
                    sonnet = []
                    sonnetToAppend = False
                continue
            sonnet.append([re.sub(r'[^\w\s\']', '', w) for
                           w in line.strip().split(' ')])
        sonnets.append(sonnet)
    f.close()
    return sonnets


def loadMoraesPoems():
    poems = []
    with codecs.open('./moraes.txt', 'r', encoding='utf-8') as f:
        poem = []
        poemToAppend = False
        for line in f:
            if line.strip().split(' ')[-1].isdigit():
                poemToAppend = True
                continue
            if line == '\n':
                if poemToAppend:
                    poems.append(poem)
                    poem = []
                    poemToAppend = False
                continue
            poem.append([re.sub(r'[,.:;!?]', '', w) for
                         w in line.strip().split(' ')])
        poems.append(poem)
    f.close()
    return poems


def loadKadarePoems():
    poems = []
    with codecs.open('./kadare.txt', 'r', encoding='utf-8') as f:
        poem = []
        poemToAppend = False
        for line in f:
            if line.strip().split(' ')[-1].isdigit():
                poemToAppend = True
                continue
            if line == '\n':
                if poemToAppend:
                    poems.append(poem)
                    poem = []
                    poemToAppend = False
                continue
            poem.append([re.sub(r'[,.:;!?]', '', w) for
                         w in line.strip().split(' ')])
        poems.append(poem)
    f.close()
    return poems


def getUniqueWords(sonnets):
    s = set()
    for sonnet in sonnets:
        for sentence in sonnet:
            s |= set([word.lower() for word in sentence])
    return(list(s))


def getUniqueSyllables(sonnets):
    h = Hyphenator('en_GB')
    s = set()
    for sonnet in sonnets:
        for sentence in sonnet:
            for word in sentence:
                syllables = h.syllables(unicode(word.lower()))
                if len(syllables) < 2:
                    s.add(unicode(word.lower()))
                else:
                    s |= set(syllables)
    return(list(s))


def getUniqueSentences(sonnets):
    s = []
    for sonnet in sonnets:
        for sentence in sonnet:
            if not sentence in s:
                s.append(sentence)
    return s


def getRhymePairs(sonnets):
    rhymes = []
    for sonnet in sonnets:
        if len(sonnet) == 14:
            rhymes.append((sonnet[0][-1], sonnet[2][-1]))
            rhymes.append((sonnet[1][-1], sonnet[3][-1]))
            rhymes.append((sonnet[4][-1], sonnet[6][-1]))
            rhymes.append((sonnet[5][-1], sonnet[7][-1]))
            rhymes.append((sonnet[8][-1], sonnet[10][-1]))
            rhymes.append((sonnet[9][-1], sonnet[11][-1]))
            rhymes.append((sonnet[12][-1], sonnet[13][-1]))
    return rhymes


def getSentenceSyllCount(sentence):
    h = Hyphenator('en_GB')
    count = 0
    for word in sentence:
        count += max(len(h.syllables(unicode(word))), 1)
    return count


def writeSonnetToTxt(sonnet):
    if not os.path.isdir(os.getcwd() + '/sonnets/'):
        os.mkdir(os.getcwd() + '/sonnets/')
    fileName = os.getcwd() + '/sonnets/' + str(datetime.datetime.now())

    with codecs.open(fileName, 'w', encoding='utf-8') as f:
        for line in sonnet:
            for word in line:
                f.write(word + ' ')
            f.write('\n')
    f.close()
