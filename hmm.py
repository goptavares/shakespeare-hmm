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
        self.numStates = numStates + 1  # add 1 for the initial state
        self.numObs = numObs
        self.transitions = np.zeros((self.numStates, self.numStates))
        self.emissions = np.zeros((self.numStates, numObs))

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
        for it in xrange(10):
            probStates = []
            probPairs = []
            # Iterate over the data points.
            for seq, i in zip(sequences, xrange(len(sequences))):
                # Expectation (E) step.
                # Get alphas using forward algorithm.
                alphas = self.forward(seq)

                # Get betas using backwards algorithm.
                betas = self.backwards(seq)

                # Compute marginal probabilities using alphas and betas.
                probStates[i] = self.computeMarginalProbPerState(alphas, betas)
                probPairs[i] = self.computeMarginalProbPerStatePair(
                    seq, alphas, betas)

            # Maximization (M) step.
            # Use marginal probabilities to update A and O matrices.
            # TODO

            # Save A and O matrices.
            self.transitions = A
            self.emissions = O

            # Check for convergence in A and O matrices.
            # TODO


    def forward(self, seq):
        """
        Args:
          seq: list corresponding to sequence of observed outputs.
        Returns:
          Matrix where each row corresponds to the alpha vector for one state.
        """
        lenSeq = len(seq)
        alphas = np.zeros((self.numStates, lenSeq))
        alphas[0,:] = 0
        alphas[:,0] = 0
        alphas[0,0] = 1
        alphas[1:,[1]] = self.transitions[[0],:] * self.emissions[:, [seq[0]]]

        # Iterate over all items in the sequence (forward) and all states.
        for t in xrange(2, lenSeq):
            for s in xrange(1, self.numStates):
                alphas[s,t] = (sum(alphas[:,[t-1]] * self.transitions[1:,[s]]) *
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
        betas[0,:] = 0
        betas[:,0] = 0
        betas[1:,-1] = 1

        # Iterate over all items in the sequence (backwards) and all states.
        for t in xrange(lenSeq-2, -1, -1):
            for s in xrange(1, self.numStates):
                betas[s,t] = sum(betas[:,[t+1]] * self.transitions[[s],1:] * 
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
        probs = []
        for j in xrange(np.shape(alphas)[1]):
            probs[j] = []
            s = 0
            for a in xrange(self.numStates):
                probs[j][a] = alphas[a,j] * betas[a,j]
                s += probs[j][a]
            # Normalize.
            for a in xrange(self.numStates):
                if s != 0:
                    probs[j][a] = probs[j][a] / float(s)
        return probs

    def computeMarginalProbPerStatePair(self, seq, alphas, betas):
        """
        Args:
          seq: list corresponding to sequence of observed outputs.
          alphas: matrix where each row corresponds to the alpha vector for one
              state.
          betas: matrix where each row corresponds to the beta vector for one
              state.
        """
        probs = []
        probs[0] = []
        for j in xrange(1, np.shape(alphas)[1]):
            probs[j] = []
            s = 0
            for a in xrange(self.numStates):
                for b in xrange(self.numStates):
                    probs[j][a,b] = (alphas[a,j-1] * self.transitions[a,b] *
                                       betas[b,j] * self.emissions[b,seq[j-1]])
                    s += probs[j][a,b]
            # Normalize.
            for a in xrange(self.numStates):
                for b in xrange(self.numStates):
                    if s != 0:
                        probs[j][a,b] = probs[j][a,b] / float(s)
        return probs

    def generateSonnet(self):
        return 0
