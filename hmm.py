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
        self.transitions = np.zeros((numStates, numStates))
        self.emissions = np.zeros((numStates, numObs))

    def train(self, sequences):
        """
        Trains the hidden Markov model on a unlabeled dataset (unsupervised),
        using the Expectation-Maximization (EM) algorithm. Updates the
        tranistion and emission matrices.
        Args:
          sequences: list where each item is a list corresponding to an observed
              sequence.
        """
        # Randomly initialize A and O matrices, making sure each row adds to 1.
        A = np.random.uniform(0, 1, (self.numStates, self.numStates))
        for i in np.shape(A)[0]:
            A[[i],:] = A[[i],:] / sum(A[[i],:])
        O = np.random.uniform(0, 1, (self.numStates, self.numObs))
        for i in np.shape(O)[0]:
            O[[i],:] = O[[i],:] / sum(O[[i],:])

        # EM algorithm.
        for i in xrange(10):
            # Iterate over the data points.
            for seq in sequences:
                # Expectation (E) step.
                # Get alphas using forward algorithm.
                alphas = self.forward(seq)

                # Get betas using backwards algorithm.
                betas = self.backwards(seq)

                # Compute marginal probabilities using alphas and betas.
                probs = self.computeMarginalProbabilities(alphas, betas)

                # Maximization (M) step.
                # Use marginal probabilities to update A and O matrices.

            # Check for convergence in A and O matrices.

        # Save A and O matrices.
        self.transitions = A
        self.emissions = O

    def forward(self, seq):
        """
        Args:
          seq: list corresponding to sequence of observed outputs.
        Returns:
          Matrix where each row corresponds to the alpha vector for one state.
        """
        lenSeq = len(seq)
        alphas = np.zeros((self.numStates, lenSeq))
        alphas[:,[0]] = self.transitions[:,[0]] * self.emissions[:, [seq[0]]]

        # Iterate over all items in the sequence (forward) and all states.
        for t in xrange(1, lenSeq):
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
        lenSeq = len(seq)
        betas = np.zeros((self.numStates, lenSeq))
        betas[:,-1] = 1

        # Iterate over all items in the sequence (backwards) and all states.
        for t in xrange(lenSeq-2, -1, -1):
            for s in xrange(self.numStates):
                betas[s,t] = sum(betas[:,[t+1]] * self.transitions[[s],:] * 
                                 self.emissions[:, [seq[t+1]]])
        return betas

    def computeMarginalProbabilities(self, alphas, betas):
        """
        Args:
          alphas: matrix where each row corresponds to the alpha vector for one
              state.
          betas: matrix where each row corresponds to the beta vector for one
              state.
        """
        return 0

    def generateSonnet(self):
        return 0
