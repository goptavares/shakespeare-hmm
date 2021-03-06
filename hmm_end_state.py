#!/usr/bin/python

"""
hmm_end_state.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import copy
import numpy as np


class HMM:
    def __init__(self, numStates, numObs):
        """
        Creates a hidden Markov model (HMM) with numStates possible states and
        numObs possible observations. The HMM is defined by a transition matrix
        A, an emission matrix B and an initialization matrix I, which are all
        initialized to zeros.
        Args:
          numStates: number of possible states.
          numObs: number of possible observations.
        """
        self.numStates = numStates + 1
        self.numObs = numObs + 1
        self.endToken = self.numObs - 1
        self.endState = self.numStates - 1
        self.A = np.zeros((self.numStates, self.numStates))
        self.B = np.zeros((self.numStates, self.numObs))
        self.I = np.zeros((self.numStates, 1))

    def train(self, sequences, maxIter=1000, threshold=0.00001,
              randomInit=True):
        """
        Trains the hidden Markov model on an unlabeled dataset (unsupervised),
        using the Expectation-Maximization (EM) algorithm. Updates the
        transition (A), emission (B) and initialization (I) matrices.
        Args:
          sequences: list where each item is a list corresponding to an observed
              sequence.
          maxIter: maximum number of algorithm iterations.
          threshold: threshold used to check for convergence in the matrices.
        """
        # Randomly initialize A, B and I matrices.
        if randomInit:
            self.A = np.random.uniform(0, 1, (self.numStates, self.numStates))
            self.A[-1,:] = 0
            for i in xrange(self.numStates):
                if sum(self.A[i,:]) != 0:
                    self.A[i,:] = self.A[i,:] / float(sum(self.A[i,:]))
            self.B = np.random.uniform(0, 1, (self.numStates, self.numObs))
            self.B[:,-1] = 0
            self.B[-1,:] = 0
            self.B[-1,-1] = 1
            for i in xrange(self.numStates):
                self.B[i,:] = self.B[i,:] / float(sum(self.B[i,:]))
            self.I = np.random.uniform(0, 1, (self.numStates, 1))
            self.I[-1] = 0
            self.I[:] = self.I[:] / float(sum(self.I[:]))

        # Append end token to every sequence.
        for seq in sequences:
            seq.append(self.endToken)

        # EM algorithm.
        for it in xrange(maxIter):
            print("Iteration " + str(it) + "...")

            gammas = []
            xis = []
            # Iterate over the data points.
            for seq in sequences:
                # Expectation (E) step.
                # Get alphas using forward algorithm.
                alphas = self.forward(seq)

                # Get betas using backwards algorithm.
                betas = self.backwards(seq)

                # Compute marginal probabilities using alphas and betas.
                gammas.append(self.computeMarginalProbsPerState(alphas, betas))
                xis.append(self.computeMarginalProbsPerStatePair(seq, alphas,
                                                                 betas))

            # Save old values of A, B and I before updating.
            prevA = copy.copy(self.A)
            prevB = copy.copy(self.B)
            prevI = copy.copy(self.I)

            # Maximization (M) step.
            # Use marginal probabilities to update A matrix.
            for i in xrange(self.numStates):
                for j in xrange(self.numStates):
                    sumNum = 0
                    sumDen = 0
                    for seq, n in zip(sequences, xrange(len(sequences))):
                        for t in xrange(len(seq) - 1):
                            sumNum += xis[n][t][i,j]
                            sumDen += gammas[n][t][i]
                    if sumDen != 0:
                        self.A[i,j] = float(sumNum) / float(sumDen)

            # Use marginal probabilities to update B matrix.
            for i in xrange(self.numStates):
                sumDen = 0
                for seq, n in zip(sequences, xrange(len(sequences))):
                    for t in xrange(len(seq)):
                        sumDen += gammas[n][t][i]
                for w in xrange(self.numObs):
                    sumNum = 0
                    for seq, n in zip(sequences, xrange(len(sequences))):
                        for t in xrange(len(seq)):
                            if seq[t] == w:
                                sumNum += gammas[n][t][i]
                    self.B[i,w] = float(sumNum) / float(sumDen)

            # Use marginal probabilities to update I matrix.
            for i in xrange(self.numStates):
                self.I[i,0] = sum([gammas[n][0][i] for n in
                                   xrange(len(sequences))])
            self.I[:] = self.I[:] / float(sum(self.I[:]))

            # Check for convergence in A, B and I matrices.
            diffA = np.absolute(np.linalg.norm(self.A, ord='fro') -
                                np.linalg.norm(prevA, ord='fro'))
            diffB = np.absolute(np.linalg.norm(self.B, ord='fro') -
                                np.linalg.norm(prevB, ord='fro'))
            diffI = np.absolute(np.linalg.norm(self.I, ord='fro') -
                                np.linalg.norm(prevI, ord='fro'))
            print("Diff A: " + str(diffA))
            print("Diff B: " + str(diffB))
            print("Diff I: " + str(diffI))
            if diffA + diffB + diffI < threshold:
                break

    def forward(self, seq):
        """
        Args:
          seq: list corresponding to sequence of observed outputs.
        Returns:
          Matrix where each row corresponds to the alpha vector for one state.
        """
        alphas = np.zeros((self.numStates, len(seq)))
        alphas[:,[0]] = self.I * self.B[:, [seq[0]]]

        # Iterate over all items in the sequence (forward) and all states.
        for t in xrange(1, len(seq)):
            for s in xrange(self.numStates):
                alphas[s,t] = (sum(alphas[:,[t-1]] * self.A[:,[s]]) *
                               self.B[s, [seq[t]]])
        return alphas

    def backwards(self, seq):
        """
        Args:
          seq: list corresponding to sequence of observed outputs.
        Returns:
          Matrix where each row corresponds to the beta vector for one state.
        """
        betas = np.zeros((self.numStates, len(seq)))
        betas[:,-1] = 1

        # Iterate over all items in the sequence (backwards) and all states.
        for t in xrange(len(seq)-2, -1, -1):
            for s in xrange(self.numStates):
                betas[s,t] = sum(betas[:,[t+1]] * np.transpose(self.A[[s],:]) *
                                 self.B[:, [seq[t+1]]])
        return betas

    def computeMarginalProbsPerState(self, alphas, betas):
        """
        Args:
          alphas: matrix where each row corresponds to the alpha vector for one
              state.
          betas: matrix where each row corresponds to the beta vector for one
              state.
        """
        gammas = []
        for t in xrange(np.shape(alphas)[1]):
            gammas.append(np.zeros((self.numStates, 1)))
            s = 0
            for i in xrange(self.numStates):
                gammas[t][i] = alphas[i,t] * betas[i,t]
                s += gammas[t][i]
            gammas[t] = gammas[t] / float(s)
        return gammas

    def computeMarginalProbsPerStatePair(self, seq, alphas, betas):
        """
        Args:
          seq: list corresponding to sequence of observed outputs.
          alphas: matrix where each row corresponds to the alpha vector for one
              state.
          betas: matrix where each row corresponds to the beta vector for one
              state.
        """
        den = sum(alphas[:,-1])
        xis = []
        for t in xrange(np.shape(alphas)[1] - 1):
            xis.append(np.zeros((self.numStates, self.numStates)))
            for i in xrange(self.numStates):
                for j in xrange(self.numStates):
                    xis[t][i,j] = (alphas[i,t] * self.A[i,j] *
                                   betas[j,t+1] * self.B[j,seq[t+1]] /
                                   float(den))
        return xis

    def generateSonnetFromWords(self, numSentences):
        """
        Args:
          numSentences: number of sentences in the sonnet to be generated.
        """
        sonnet = []
        for s in xrange(numSentences):
            sentence = []
            currState = np.random.choice(self.numStates, p=self.I[:,0])
            while True:
                if currState == self.endState:
                    break
                word = np.random.choice(self.numObs, p=self.B[currState,:])
                sentence.append(word)
                currState = np.random.choice(self.numStates,
                                             p=self.A[currState,:])
            sonnet.append(sentence)
        return sonnet

    def generateSonnetFromSyllables(self, numSentences, numWordsPerSentence,
                                    validWords=None, tokens=None):
        sonnet = []
        for s in xrange(numSentences):
            sentence = []
            w = 0
            while w < numWordsPerSentence:
                word = []
                currState = np.random.choice(self.numStates, p=self.I[:,0])
                while True:
                    if currState == self.endState:
                        break
                    syllable = np.random.choice(self.numObs,
                                                p=self.B[currState,:])
                    word.append(syllable)
                    currState = np.random.choice(self.numStates,
                                                 p=self.A[currState,:])
                if validWords:
                    untokenizedWord = ''
                    for syll in word:
                        untokenizedWord += str(tokens[syll])
                    if untokenizedWord in validWords:
                        sentence.append(word)
                        w += 1
                    else:
                        continue
                else:
                    sentence.append(word)
                    w += 1
            sonnet.append(sentence)
        return sonnet

    def generateSonnetFromSentences(self):
        """
        Args:
          numSentences: number of sentences in the sonnet to be generated.
        """
        sonnet = []
        currState = np.random.choice(self.numStates, p=self.I[:,0])
        while True:
            if currState == self.endState:
                break
            sentence = np.random.choice(self.numObs, p=self.B[currState,:])
            sonnet.append(sentence)
            currState = np.random.choice(self.numStates, p=self.A[currState,:])
        return sonnet

    def generateSonnetWithRhymes(self, rhymePairs, numSentences=14):
        lastWords = []
        for s in xrange((numSentences-2) / 4):
            rhymeIdx = np.random.choice(len(rhymePairs), 2)
            lastWords += [rhymePairs[rhymeIdx[0]][0],
                          rhymePairs[rhymeIdx[1]][0],
                          rhymePairs[rhymeIdx[0]][1],
                          rhymePairs[rhymeIdx[1]][1]]
        rhymeIdx = np.random.choice(len(rhymePairs))
        lastWords += [rhymePairs[rhymeIdx][0],
                      rhymePairs[rhymeIdx][1]]

        sonnet = []
        for lastWord in lastWords:
            sentence = [lastWord]
            lastState = np.argmax(self.B[:, lastWord])
            currState = np.random.choice(self.numStates,
                                         p=self.A[lastState,:])
            while True:
                if currState == self.endState:
                    break
                word = np.random.choice(self.numObs, p=self.B[currState,:])
                sentence.append(word)
                currState = np.random.choice(self.numStates,
                                             p=self.A[currState,:])
            sonnet.append(sentence)

        return [sentence[::-1] for sentence in sonnet]
