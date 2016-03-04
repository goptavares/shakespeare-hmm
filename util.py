#!/usr/bin/python

"""
util.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import datetime
import os
import re


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


def writeSonnetToTxt(sonnet):
    if not os.path.isdir(os.getcwd() + '/sonnets/'):
        os.mkdir(os.getcwd() + '/sonnets/')
    fileName = os.getcwd() + '/sonnets/' + str(datetime.datetime.now())

    with open(fileName, 'w') as f:
        for line in sonnet:
            for word in line:
                f.write(word + ' ')
            f.write('\n')
    f.close()
