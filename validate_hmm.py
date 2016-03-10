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


def main():
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

if __name__ == '__main__':
    main()