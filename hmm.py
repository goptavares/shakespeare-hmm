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

	def train(self, seq):
		"""
		Trains the hidden Markov model on a unlabeled dataset (unsupervised),
		using the Expectation-Maximization (EM) algorithm. Updates the
		tranistion and emission matrices.
		Args:
	      seq: list corresponding to sequence of observed outputs.
		"""
		# Randomly initialize A and O matrices.
		A = np.random.normal(0, 1, (self.numStates, self.numStates))
		O = np.random.normal(0, 1, (self.numStates, self.numObs))

		# EM algorithm.
		for i in xrange(10):
			# Expectation step.
			# Get alphas using forward algorithm.
			alphas = self.forward(seq)

			# Get betas using backwards algorithm.
			betas = self.backwards(seq)

			# Compute marginal probabilities using alphas and betas.
			probs = self.computeMarginalProbabilities(alphas, betas)

			# Maximization step.
			# Use marginal probabilities to update A and O matrices.

			# Check for convergence.

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
	    lenSeq = len(seq)  # length of sequence
	    alphas = np.zeros((self.numStates, lenSeq))
	    alphas[:,0] = self.transitions[:,1] * self.emissions[:, seq[0]]

	    # Iterate over all items in the sequence and all states.
	    for t in xrange(1, lenSeq):
	        for s in xrange(self.numStates):
	            alphas[s,t] = (sum(alphas[:,t-1] * self.transitions[:,s]) *
	            			   self.emissions[s, seq[t]])
	    return alphas

	def backwards(self, seq):
		"""
	    Args:
	      seq: list corresponding to sequence of observed outputs.
	    Returns:
	      Matrix where each row corresponds to the beta vector for one state.
	    """
	    lenSeq = len(seq)  # length of sequence
		betas = np.zeros((self.numStates, lenSeq))
		betas[:,-1] = np.ones((np.shape(betas[:,-1])))

		return betas

	def computeMarginalProbabilities(self, alphas, betas):
		return 0
