#!/usr/bin/python

"""
hmm.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np

class HMM:
    def __init__(self, numStates, numObs):
        """
        Args:
          numStates: number of possible states.
          numObs: number of possible observations.
        """
        self.numStates = numStates
        self.numObs = numObs
        self.transitions = np.zeros((self.numStates, self.numStates))
        self.emissions = np.zeros((self.numStates, self.numObs))
        self.initial = np.zeros((self.numStates, 1))

    def train(self, sequences):
        """
        Trains the hidden Markov model on a unlabeled dataset (unsupervised),
        using the Expectation-Maximization (EM) algorithm. Updates the
        tranistion and emission matrices.
        Args:
          sequences: list where each item is a list corresponding to an observed
              sequence.
        """
        # Randomly initialize A, B and I matrices.
        A = np.random.uniform(0, 1, (self.numStates, self.numStates))
        for i in xrange(self.numStates):
            A[i,:] = A[i,:] / float(sum(A[i,:]))
        B = np.random.uniform(0, 1, (self.numStates, self.numObs))
        for i in xrange(self.numStates):
            B[i,:] = B[i,:] / float(sum(B[i,:]))
        I = np.random.uniform(0, 1, (self.numStates, 1))
        I[:,0] = I[:,0] / float(sum(I[:,0]))

        self.transitions = A
        self.emissions = B
        self.initial = I

        # Thresholds used to check for convergence.
        threshA = 100
        threshB = 1000
        threshI = 10

        # EM algorithm.
        for it in xrange(10):
            print("Itaration " + str(it) + "...")
            gammas = []
            xis = []
            # Iterate over the data points.
            for seq, n in zip(sequences, xrange(len(sequences))):
                # Expectation (E) step.
                # Get alphas using forward algorithm.
                alphas = self.forward(seq)

                # Get betas using backwards algorithm.
                betas = self.backwards(seq)

                # Compute marginal probabilities using alphas and betas.
                gammas.append(self.computeMarginalProbPerState(alphas, betas))
                xis.append(self.computeMarginalProbPerStatePair(seq, alphas,
                                                                betas))

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
                        A[i,j] = float(sumNum) / float(sumDen)

            # Use marginal probabilities to update B matrix.
            for i in xrange(self.numStates):
                for w in xrange(self.numObs):
                    sumNum = 0
                    sumDen = 0
                    for seq, n in zip(sequences, xrange(len(sequences))):
                        for t in xrange(len(seq)):
                            if seq[t] == w:
                                sumNum += gammas[n][t][i]
                            sumDen += gammas[n][t][i]
                    if sumDen != 0:
                        B[i,w] = float(sumNum) / float(sumDen)

            # Use marginal probabilities to update I matrix.
            for i in xrange(self.numStates):
                I[i,0] = sum([gammas[n][0][i] for n in xrange(len(sequences))])
            for i in xrange(self.numStates):
                I[i,0] = I[i,0] / float(sum(I[:,0]))

            # Save A, B and I matrices.
            prevA = self.transitions
            prevB = self.emissions
            prevI = self.initial
            self.transitions = A
            self.emissions = B
            self.initial = I

            # Check for convergence in A, B and I matrices.
            diffA = np.linalg.norm(A, ord=2) - np.linalg.norm(prevA, ord=2)
            diffB = np.linalg.norm(B, ord=2) - np.linalg.norm(prevB, ord=2)
            diffI = np.linalg.norm(I, ord=2) - np.linalg.norm(prevI, ord=2)
            print("Diff A: " + str(diffA))
            print("Diff B: " + str(diffB))
            print("Diff I: " + str(diffI))
            if diffA < threshA and diffB < threshB and diffI < threshI:
                break

    def forward(self, seq):
        """
        Args:
          seq: list corresponding to sequence of observed outputs.
        Returns:
          Matrix where each row corresponds to the alpha vector for one state.
        """
        alphas = np.zeros((self.numStates, len(seq)))
        alphas[:,[0]] = self.initial * self.emissions[:, [seq[0]]]

        # Iterate over all items in the sequence (forward) and all states.
        for t in xrange(1, len(seq)):
            for s in xrange(self.numStates):
                alphas[s,t] = (sum(alphas[:,[t-1]] * self.transitions[:,[s]]) *
                               self.emissions[s, [seq[t]]])
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
                betas[s,t] = sum(betas[:,[t+1]] *
                                 np.transpose(self.transitions[[s],:]) * 
                                 self.emissions[:, [seq[t+1]]])
        return betas

    def computeMarginalProbPerState(self, alphas, betas):
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
            # Normalize.
            for i in xrange(self.numStates):
                if s != 0:
                    gammas[t][i] = gammas[t][i] / float(s)
        return gammas

    def computeMarginalProbPerStatePair(self, seq, alphas, betas):
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
                    xis[t][i,j] = (alphas[i,t] * self.transitions[i,j] *
                                   betas[j,t+1] * self.emissions[j,seq[t+1]] /
                                   float(den))
        return xis

    def generateSonnet(self):
        return 0
