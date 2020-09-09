#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 17:01:04 2020

@author: duilio_97
"""

import numpy as np
from collections import deque
from itertools import product
from collections import OrderedDict
from operator import itemgetter
import matplotlib.pyplot as plt
  
def partition(filename, dlm, l):
    '''
    Detects communities in a correlation matrix by using a spectral optimization process
    
    See:

    MacMahon, M. and Garlaschelli, D. (2015). Community Detection for Correlation Matrices.
    Phys. Rev. X, 5(2), 021006. https://link.aps.org/doi/10.1103/PhysRevX.5.021006
        
    The following python implementation clusters a network into several modules
    using modularity maximization by spectral methods and has been used as a reference:
    
    Zuo, Z. (2018). python-modularity-maximization
    (Available at https://github.com/zhiyzuo/python-modularity-maximization)

    Parameters
    ----------
    filename: str
        The name of the file containing the time series of interest
    dlm: str
        The string used to separate values in 'filename'
    l: int
        Selects the null model, following the notation used in the reference paper (Eq. 37)

    Returns
    -------
    dict
        A dictionary that saves membership
        Key: vertex name; Value: community index
    '''
    
    global N
    
    f = open(filename, "r")
    # Reads the time series
    # Each column corresponds to a vertex (e.g., a stock)
    data = np.loadtxt(filename, delimiter = dlm, skiprows = 1)
    N = len(data[0])
    n_verts = np.arange(1, N + 1)
    # Reads the names of the vertices (header)
    firstline = f.readline()
    firstline = firstline.rstrip('\n')
    names_list = firstline.split(dlm)
    names_dict = dict(zip(n_verts, names_list))
    f.close()

    # Computes the modularity matrix for the entire data set
    B = get_base_modularity_matrix_shuffle(data, l)
    # B = get_base_modularity_matrix_MP(data, l)
    
    # Initial community is potentially divisible
    divisible_comm = deque([0])
    # All vertices as one group
    comm_dict = {u: 0 for u in n_verts}
    # Initializes the community counter
    comm_counter = 0
    
    while len(divisible_comm) > 0:
        
        # Gets out the index of the potentially divisible community
        comm_index = divisible_comm.popleft()
        g1_verts, comm_verts = divide(comm_dict, comm_index, B)
        
        if g1_verts is None:
            # Indivisible, go to the next one
            continue
        
        # If divisibile, obtains the other group g2
        g2_verts = sorted(set(comm_verts).difference(set(g1_verts)))
        
        comm_counter += 1
        # A further division of the group will be attempted
        divisible_comm.append(comm_counter)
        # Updates the community
        for u in g1_verts:
            comm_dict[u] = comm_counter
        
        comm_counter += 1
        divisible_comm.append(comm_counter)
        for u in g2_verts:
            comm_dict[u] = comm_counter     
    
    # visual_data(data, names_list)
    visual_ord_data(data, comm_dict, names_dict)
    
    return {names_dict[u]: comm_dict[u] for u in n_verts}

def get_base_modularity_matrix_shuffle(data, l):
    '''
    Computes the modularity matrix for the whole data set
    (randomly shuffles the original time series to evaluate the 'random' component)

    Parameters
    ----------
    data: np.matrix
        The complete matrix of the time series
    l: int
        Selects the null model, following the notation used in the reference paper (Eq. 37)

    Returns
    -------
    np.matrix
        The modularity matrix for 'data'
    '''
    
    global N, C_norm
    
    # Computes the correlation matrix (+ the sum of all its elements, 
    # its eigenvalues and eigenvectors)
    C = np.corrcoef(data, rowvar = False)
    C_norm = np.sum(C)
    evals, evecs = np.linalg.eigh(C)
    
    if l == 1:
        I = np.identity(N)
        
        return C - I
    
    elif l == 2:
        
        # Randomly shuffles each one of the time series
        data_shuffle = np.copy(data)
        for i in range(N):
            np.random.shuffle(data_shuffle[:, i])
        
        # Computes the correlation matrix for the shuffled series (+ its eigenvalues and eigenvectors)
        C_shuffle = np.corrcoef(data_shuffle, rowvar = False)
        evals_shuffle, evecs_shuffle = np.linalg.eigh(C_shuffle)
        
        # Evaluates the 'random' component
        j = np.argmax(evals_shuffle)
        k = np.argmin(evals_shuffle)
        set_r = np.argwhere((evals <= evals_shuffle[j]) & (evals >= evals_shuffle[k]))
        C_r = np.zeros(shape = (N, N))
        for l in set_r:
            C_r += evals[l]*np.outer(evecs[:, l], evecs[:, l])
        
        return C - C_r
    
    elif l == 3:
        
        # Randomly shuffles each one of the time series
        data_shuffle = np.copy(data)
        for i in range(N):
            np.random.shuffle(data_shuffle[:, i])
        
        # Computes the correlation matrix for the shuffled series (+ its eigenvalues and eigenvectors)
        C_shuffle = np.corrcoef(data_shuffle, rowvar = False)
        evals_shuffle, evecs_shuffle = np.linalg.eigh(C_shuffle)
        
        # Evaluates the 'random' component
        j = np.argmax(evals_shuffle)
        k = np.argmin(evals_shuffle)
        set_r = np.argwhere((evals <= evals_shuffle[j]) & (evals >= evals_shuffle[k]))
        C_r = np.zeros(shape = (N, N))
        for l in set_r:
            C_r += evals[l]*np.outer(evecs[:, l], evecs[:, l])
        
        # Evaluates the 'market' mode
        m = np.argmax(evals)
        C_m = evals[m]*np.outer(evecs[:, m], evecs[:, m])
        
        return C - C_r - C_m
    
    else:
        print("The variable 'l' can only assume the values 1, 2 or 3")
        quit()

def get_base_modularity_matrix_MP(data, l):
    '''
    Computes the modularity matrix for the whole data set
    (uses the Marchenko-Pastur distribution to evaluate the 'random' component)

    Parameters
    ----------
    data: np.matrix
        The complete matrix of the time series
    l: int
        Selects the null model, following the notation used in the reference paper (Eq. 37)

    Returns
    -------
    np.matrix
        The modularity matrix for 'data'
    '''
    
    global N, C_norm
    
    # Calculates the max and min eigenvalues of the Marchenko-Pastur distribution
    T = len(data)
    lambda_max = (1 + np.sqrt(N/T))**2
    lambda_min = (1 - np.sqrt(N/T))**2
    
    # Computes the correlation matrix (+ the sum of all its elements, 
    # its eigenvalues and eigenvectors)
    C = np.corrcoef(data, rowvar = False)
    C_norm = np.sum(C)
    evals, evecs = np.linalg.eigh(C)
    
    if l == 1:
        I = np.identity(N)
        
        return C - I
    
    elif l == 2:
        # Evaluates the 'random' component
        set_r = np.argwhere((evals <= lambda_max) & (evals >= lambda_min))
        C_r = np.zeros(shape = (N, N))
        for i in set_r:
            C_r += evals[i]*np.outer(evecs[:, i], evecs[:, i])
        
        return C - C_r
    
    elif l == 3:
        
        # Evaluates the 'random' component
        set_r = np.argwhere((evals <= lambda_max) & (evals >= lambda_min))
        C_r = np.zeros(shape = (N, N))
        for i in set_r:
            C_r += evals[i]*np.outer(evecs[:, i], evecs[:, i])
        
        # Evaluates the 'market' mode
        j = np.argmax(evals)
        C_m = evals[j]*np.outer(evecs[:, j], evecs[:, j])
        
        return C - C_r - C_m
    
    else:
        print("The variable 'l' can only assume the values 1, 2 or 3")
        quit()

def divide(comm_dict, comm_index, B):
    '''
    Bisection of a community in 'data'

    Parameters
    ----------
    comm_dict: dict
        A dictionary to store the membership of each vertex
    comm_index: int
        The number that identifies the community of interest
    B: np.matrix
        The modularity matrix for the whole data set

    Returns
    -------
    tuple
        If the given community is indivisible, returns (None, None)
        If the given community is divisible, returns a tuple where
        the 1st element is a np.array for the 1st sub-group and
        the 2nd element is a tuple for the original group
    '''
    
    global N
    
    # Original group
    comm_verts = tuple(u for u in comm_dict if comm_dict[u] == comm_index)
    
    if len(comm_verts) == 1:
        # The group consists of a single vertex, therefore it is indivisible
        return None, None
    
    if len(comm_verts) == N:
        # This holds for the initial community (0)
        C_A = B
    else:
        C_A = get_modularity_matrix(comm_verts, B)
    
    # Computes the eigenvalues and eigenvectors of the modularity matrix
    evals, evecs = np.linalg.eigh(C_A)
    
    # Finds the largest eigenvalue (and the corresponding eigenvector) of 'C_A'
    j = np.argmax(evals)
    mx_evals = evals[j]
    mx_evecs = evecs[:, j]
    
    if mx_evals > 0:
        
        # Constructs the vector matching the signs of the components of the 
        # eigenvector corresponding to the largest eigenvalue of 'C_A'
        s = np.ones(shape = len(mx_evecs), dtype = int)
        k = np.argwhere(mx_evecs < 0)
        s[k] = -1
        
        delta_modularity = get_delta_Q(C_A, s)

        if delta_modularity > 0:
        
            g1_verts = np.array([comm_verts[i] for i in range(mx_evecs.shape[0]) if s[i] > 0])
            
            if len(g1_verts) == len(comm_verts) or len(g1_verts) == 0:
                # Indivisible
                return None, None
            # Divisible
            return g1_verts, comm_verts
      
    # Indivisible
    return None, None

def get_modularity_matrix(comm_verts, B):
    '''
    Computes the modularity matrix for a specific group (generalized modularity matrix)

    Parameters
    ----------
    comm_verts: tuple
        The group of interest
    B: np.matrix
        The modularity matrix for the whole data set

    Returns
    -------
    np.matrix
        The modularity of 'comm_verts' within 'data'
    '''
    
    indices = tuple(u - 1 for u in comm_verts)
    C_A = B[indices, :][:, indices]
    
    return C_A

def get_delta_Q(C_A, s):
    '''
    Calculates the modularity change according to Eq. A16 in the reference paper

    Parameters
    ----------
    C_A: np.matrix
        The generalized modularity matrix
    s: np.array
        The membership vector

    Returns
    -------
    float
        The corresponding modularity change
    '''
    
    global C_norm
    
    C_AA = np.sum(C_A)
    delta_Q = ((s.T.dot(C_A)).dot(s) - C_AA)/(2.*C_norm)
    
    return delta_Q

def get_modularity(B, comm_dict, n_verts):
    '''
    Calculates the modularity according to Eq. 36 in the reference paper

    Parameters
    ----------
    B: np.matrix
        The modularity matrix for the whole data set
    comm_dict: dict
        A dictionary to store the membership of each vertex
    n_verts: np.array
        np.arange(1, N + 1) 

    Returns
    -------
    float
        The modularity of 'data' given 'comm_dict'
    '''
    
    global N, C_norm
    
    foo = np.sum([B[i, j] for i, j in product(range(N), range(N)) \
                if comm_dict[n_verts[i]] == comm_dict[n_verts[j]]])
    Q = foo/C_norm
    
    return Q

def visual_data(data, names_list):
    '''
    Saves the heatmap showing the values of the entries of the correlation matrix

    Parameters
    ----------
    data: np.matrix
        The complete matrix of the time series
    names_list: list
        The names of the vertices
    '''
    
    global N

    C = np.corrcoef(data, rowvar = False)
    
    fig, ax = plt.subplots()
    im = ax.imshow(C, cmap = 'RdBu_r')
    im.set_clim(-1, 1)
    ax.figure.colorbar(im, ax = ax)
    # ax.set_xticks(np.arange(N))
    # ax.set_yticks(np.arange(N))
    # ax.set_xticklabels(names_list)
    # ax.set_yticklabels(names_list)
    # plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")
    plt.savefig("spec.png", dpi = 1000)
    plt.close()

def visual_ord_data(data, comm_dict, names_dict):
    '''
    Using the membership dictionary, orders 'data' and saves the heatmap showing
    the values of the entries of the ordered correlation matrix

    Parameters
    ----------
    data: np.matrix
        The complete matrix of the time series
    comm_dict: dict
        A dictionary to store the membership of each vertex
    names_dict: dict
        A dictionary to store the names of the vertices
    '''
    
    global N
    
    ord_dict = OrderedDict(sorted(comm_dict.items(), key = itemgetter(1)))
    ord_dict1 = {i - 1: ord_dict[i] for i in ord_dict.keys()}
    ord_names = {i: names_dict[i] for i in ord_dict.keys()}
    ord_data = np.column_stack([data[:, i] for i in ord_dict1.keys()])
    # Alternative to np.column_stack:
    # keys_array = np.array(list(ord_dict1.keys()))
    # ord_data = data[:, keys_array]
    
    ord_C = np.corrcoef(ord_data, rowvar = False)
    
    fig, ax = plt.subplots()
    im = ax.imshow(ord_C, cmap = 'RdBu_r')
    im.set_clim(-1, 1)
    ax.figure.colorbar(im, ax = ax)
    # ax.set_xticks(np.arange(N))
    # ax.set_yticks(np.arange(N))
    # ax.set_xticklabels(ord_names.values())
    # ax.set_yticklabels(ord_names.values())
    # plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")
    plt.savefig("spec.png", dpi = 1000)
    plt.close()