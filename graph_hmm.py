#!/usr/bin/python

"""
graph_hmm.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""

import networkx as ntx
import matplotlib.pyplot as plt
import numpy as np
import util
import copy

def graphHMM(hmmModel):
    A = copy.copy(hmmModel.A)
    B = copy.copy(hmmModel.B)
    I = copy.copy(hmmModel.I)
    numStates = hmmModel.numStates
    numObs = hmmModel.numObs
 
    # Import the list of unique words.
    sonnets = util.loadShakespeareSonnets()
    uniqueWords = util.getUniqueWords(sonnets)
    
    # Normalize the transition matrices by the frequency of the words.
    sonnets = util.loadShakespeareSonnets()
    tokens = util.getUniqueWords(sonnets)
    
    freq = {}
    for i in xrange(len(tokens)):
        freq[i] = 0
    for sonnet in sonnets:
        for sentence in sonnet:
            for word in sentence:
                freq[tokens.index(word.lower())] += 1

    # Visualize the initial state transitions.
    G = ntx.Graph(name='Words')
    G.add_nodes_from(ntx.path_graph(numStates + 1))
    
    # Add the edges to the graph representing the initial state transition.
    mapping = dict()
    weights = []
    for i in xrange(numStates):
        # For each state, get the top 5 words emitted.
        for k in xrange(np.shape(B)[1]):
            B[i,k] = B[i,k] / float(freq[k])  
              
        topStateWordsIdx = np.argsort(B[i,:])
        topStateWordsIdx = np.flipud(topStateWordsIdx)
        topStateWordsIdx = topStateWordsIdx[0:5]
        topStateWords = []
        for j in topStateWordsIdx:
            topStateWords.append(uniqueWords[j])
        mapping[i] = ','.join(topStateWords)
        G.add_edges_from([(numStates, i, {'weight': I[i,0]})])
        weights.append(I[i,0])
        
    # Add the label for the start state.
    mapping[numStates] = 'Start';
    G = ntx.relabel_nodes(G, mapping, copy=False)
    ntx.draw(G, ntx.circular_layout(G), node_color='w', width=weights)
    plt.show()
    
    
    # Visualize the state transitions as a directed graph.
    del mapping[numStates]  # remove the start state
    H = ntx.MultiDiGraph(name='Words')
    H.add_nodes_from(ntx.path_graph(numStates))
    
    weights = []    
    for i in xrange(numStates):
        for j in xrange(numStates):
            H.add_edges_from([(i, j, {'weight': A[i,j]})])
            weights.append(A[i,j])

    H = ntx.relabel_nodes(H, mapping, copy=False)
    ntx.draw(H, ntx.circular_layout(H), node_color='w', width=weights)
    plt.show()
