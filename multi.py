#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 17:01:04 2020

@author: duilio_97
"""

import numpy as np
from collections import deque
from itertools import product
import matplotlib.pyplot as plt

def multiresolution_detect(filename, dlm, l):
    '''
    Implements the multiresolution community detection approach introduced in Sec. IV C
    of the following paper:

    MacMahon, M. and Garlaschelli, D. (2015). Community Detection for Correlation Matrices.
    Phys. Rev. X, 5(2), 021006. https://link.aps.org/doi/10.1103/PhysRevX.5.021006

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
    lvls: list
        A list of dictionaries that saves membership at each iteration (level) of
        the spectral optimization process
    '''
    
    f = open(filename, "r")
    # Reads the time series
    # Each column corresponds to a vertex (e.g., a stock)
    data = np.loadtxt(filename, delimiter = dlm, skiprows = 1)
    # Reads the names of the vertices (header)
    firstline = f.readline()
    firstline = firstline.rstrip('\n')
    names_list = firstline.split(dlm)
    f.close()
    
    # Number of indivisible sets of vertices
    nondiv_count = 0
    # Labels the first level
    lvl = 1
    # This dict stores the membership of each vertex at a given level
    lvl_comm_dict = {}
    # This dict stores the indivisible sets of vertices
    next_lvl_nondiv = {}
    # Sets up the first iteration
    lvl_data = deque([data])
    lvl_names = deque([names_list]) 
    # The data sets (and the corresponding names) for the next iteration will be stored here
    next_lvl_data = deque([])
    next_lvl_names = deque([])

    indices = []
    lvls = []
    
    while len(lvl_data) > 0:
        
        lvl_divisible_comm = deque(np.ones(shape = len(lvl_data), dtype = int))
        floor = nondiv_count
              
        while len(lvl_divisible_comm) > 0:
            
            # Gets out a particular data set (and the corresponding names)
            data_set = lvl_data.popleft()
            names_set = lvl_names.popleft()
            comm_count = lvl_divisible_comm.popleft() + floor
            comm_dict, floor = partition(data_set, l, names_set, comm_count)
            
            if comm_dict is None:
                # Indivisible
                # Updates the community dictionary at the present level
                lvl_comm_dict.update({names_set[u]: comm_count for u in np.arange(len(data_set[0]))})
                # Stores this set as an indivisible one
                nondiv_count += 1
                next_lvl_nondiv.update({names_set[u]: nondiv_count for u in np.arange(len(data_set[0]))})
                continue
            
            # Divisible
            # Updates the community dictionary at the present level
            lvl_comm_dict.update(comm_dict)
            
            # At the next iteration, the spectral method will be applied 
            # to each one of the potentially divisible communities detected
            comm_labels = set(comm_dict.values())
            for i in comm_labels:
                comm_names = list(u for u in comm_dict if comm_dict[u] == i)
                if len(comm_names) == 1:
                    # Single-vertex communities are trivially indivisible
                    nondiv_count += 1
                    next_lvl_nondiv.update({comm_names[0]: nondiv_count})
                else:
                    # Saves a particular community as a data_set for the next iteration
                    for j in comm_names:
                        for k, n in enumerate(names_list):
                            if(j == n): indices.append(k)
                    next_lvl_names.append(comm_names)        
                    next_lvl_data.append(data[:, indices.copy()])
                    indices.clear()
        
        visual_data(data, lvl, lvl_comm_dict.copy(), names_list)
        
        # Updates the answer
        lvls.append(lvl_comm_dict.copy())
        # Sets up the next iteration
        lvl_comm_dict.clear()
        lvl += 1
        lvl_data = next_lvl_data
        lvl_names = next_lvl_names
        lvl_comm_dict.update(next_lvl_nondiv.copy())
        
    return lvls
    
def partition(data_set, l, names_set, index_set):
    '''
    Detects communities in a correlation matrix by using a spectral optimization process
        
    The following python implementation clusters a network into several modules
    using modularity maximization by spectral methods and has been used as a reference:
    
    Zuo, Z. (2018). python-modularity-maximization
    (Available at https://github.com/zhiyzuo/python-modularity-maximization)

    Parameters
    ----------
    data_set: np.matrix
        The set of time series under consideration
    l: int
        Selects the null model, following the notation used in the reference paper (Eq. 37)
    names_set: list
        The names of the vertices under consideration
    index_set: int
        The number that identifies the set under consideration

    Returns
    -------
    tuple
        If the given set is indivisible, returns (None, comm_counter)
        If the given set is divisible, returns a tuple where 
        the 1st element is a dictionary that saves membership and
        the 2nd element is the community counter
    '''
    
    N_set = len(data_set[0])
    verts_set = np.arange(1, N_set + 1)
    
    # Computes the modularity matrix for 'data_set'
    B, C_norm = get_base_modularity_matrix_shuffle(data_set, l)
    # B, C_norm = get_base_modularity_matrix_MP(data_set, l)
    
    # Initial community is potentially divisible
    divisible_comm = deque([index_set])
    # All the considered vertices as one group
    comm_dict = {u: index_set for u in verts_set}
    # Initializes the community counter
    comm_counter = index_set
    
    while len(divisible_comm) > 0:
        
        # Gets out the index of the potentially divisible community
        comm_index = divisible_comm.popleft()
        g1_verts, comm_verts = divide(comm_dict, comm_index, B, N_set, C_norm)
    
        if g1_verts is None:
            # Indivisible, goes to the next one
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
    
    if comm_counter == index_set:
        # Indivisible
        return None, comm_counter
    else:
        # Divisible
        return {names_set[u - 1]: comm_dict[u] for u in verts_set}, comm_counter

def get_base_modularity_matrix_shuffle(data_set, l):
    '''
    Computes the modularity matrix for 'data_set'
    (randomly shuffles the original time series to evaluate the 'random' component)

    Parameters
    ----------
    data_set: np.matrix
        The set of time series under consideration
    l: int
        Selects the null model, following the notation used in the reference paper (Eq. 37)

    Returns
    -------
    tuple
        The 1st element is the modularity matrix for 'data_set' and
        the 2nd element is the sum of all the elements of the correlation matrix
    '''
    
    N_set = len(data_set[0])
    
    # Computes the correlation matrix (+ the sum of all its elements, 
    # its eigenvalues and eigenvectors)
    C = np.corrcoef(data_set, rowvar = False)
    C_norm = np.sum(C)
    evals, evecs = np.linalg.eigh(C)
    
    if l == 1:
        I = np.identity(N_set)
        
        return C - I, C_norm
    
    elif l == 2:
        
        # Randomly shuffles each one of the time series
        data_set_shuffle = np.copy(data_set)
        for i in range(N_set):
            np.random.shuffle(data_set_shuffle[:, i])
        
        # Computes the correlation matrix for the shuffled series (+ its eigenvalues and eigenvectors)
        C_shuffle = np.corrcoef(data_set_shuffle, rowvar = False)
        evals_shuffle, evecs_shuffle = np.linalg.eigh(C_shuffle)
        
        # Evaluates the 'random' component
        j = np.argmax(evals_shuffle)
        k = np.argmin(evals_shuffle)
        set_r = np.argwhere((evals <= evals_shuffle[j]) & (evals >= evals_shuffle[k]))
        C_r = np.zeros(shape = (N_set, N_set))
        for l in set_r:
            C_r += evals[l]*np.outer(evecs[:, l], evecs[:, l])
        
        return C - C_r, C_norm
    
    elif l == 3:
        
        # Randomly shuffles each one of the time series
        data_set_shuffle = np.copy(data_set)
        for i in range(N_set):
            np.random.shuffle(data_set_shuffle[:, i])
        
        # Computes the correlation matrix for the shuffled series (+ its eigenvalues and eigenvectors)
        C_shuffle = np.corrcoef(data_set_shuffle, rowvar = False)
        evals_shuffle, evecs_shuffle = np.linalg.eigh(C_shuffle)
        
        # Evaluates the 'random' component
        j = np.argmax(evals_shuffle)
        k = np.argmin(evals_shuffle)
        set_r = np.argwhere((evals <= evals_shuffle[j]) & (evals >= evals_shuffle[k]))
        C_r = np.zeros(shape = (N_set, N_set))
        for l in set_r:
            C_r += evals[l]*np.outer(evecs[:, l], evecs[:, l])
            
        # Evaluates the 'market' mode
        m = np.argmax(evals)
        C_m = evals[m]*np.outer(evecs[:, m], evecs[:, m])
        
        return C - C_r - C_m, C_norm
    
    else:
        print("The variable 'l' can only assume the values 1, 2 or 3")
        quit()

def get_base_modularity_matrix_MP(data_set, l):
    '''
    Computes the modularity matrix for 'data_set'
    (uses the Marchenko-Pastur distribution to evaluate the 'random' component)

    Parameters
    ----------
    data_set: np.matrix
        The set of time series under consideration
    l: int
        Selects the null model, following the notation used in the reference paper (Eq. 37)

    Returns
    -------
    tuple
        The 1st element is the modularity matrix for 'data_set' and
        the 2nd element is the sum of all the elements of the correlation matrix
    '''
    
    # Calculates the max and min eigenvalues of the Marchenko-Pastur distribution
    N_set = len(data_set[0])
    T = len(data_set)
    lambda_max = (1 + np.sqrt(N_set/T))**2
    lambda_min = (1 - np.sqrt(N_set/T))**2
    
    # Computes the correlation matrix (+ the sum of all its elements, 
    # its eigenvalues and eigenvectors)
    C = np.corrcoef(data_set, rowvar = False)
    C_norm = np.sum(C)
    evals, evecs = np.linalg.eigh(C)
    
    if l == 1:
        I = np.identity(N_set)
        
        return C - I, C_norm
    
    elif l == 2:
        # Evaluates the 'random' component
        set_r = np.argwhere((evals <= lambda_max) & (evals >= lambda_min))
        C_r = np.zeros(shape = (N_set, N_set))
        for i in set_r:
            C_r += evals[i]*np.outer(evecs[:, i], evecs[:, i])
        
        return C - C_r, C_norm
    
    elif l == 3:
        
        # Evaluates the 'random' component
        set_r = np.argwhere((evals <= lambda_max) & (evals >= lambda_min))
        C_r = np.zeros(shape = (N_set, N_set))
        for i in set_r:
            C_r += evals[i]*np.outer(evecs[:, i], evecs[:, i])
            
        # Evaluates the 'market' mode
        j = np.argmax(evals)
        C_m = evals[j]*np.outer(evecs[:, j], evecs[:, j])
        
        return C - C_r - C_m, C_norm
    
    else:
        print("The variable 'l' can only assume the values 1, 2 or 3")
        quit()

def divide(comm_dict, comm_index, B, N_set, C_norm):
    '''
    Bisection of a community in 'data_set'

    Parameters
    ----------
    comm_dict: dict
        A dictionary to store the membership of each vertex in 'data_set'
    comm_index: int
        The number that identifies the community of interest
    B: np.matrix
        The modularity matrix for 'data_set'
    N_set: int
        The number of vertices in 'data_set'
    C_norm: float
        The the sum of all the elements of the correlation matrix

    Returns
    -------
    tuple
        If the given community is indivisible, returns (None, None)
        If the given community is divisible, returns a tuple where
        the 1st element is a np.array for the 1st sub-group and
        the 2nd element is a tuple for the original group
    '''
    
    # Original group
    comm_verts = tuple(u for u in comm_dict if comm_dict[u] == comm_index)
    
    if len(comm_verts) == 1:
        # The group consists of a single vertex, therefore it is indivisible
        return None, None
    
    if len(comm_verts) == N_set:
        # This holds for the initial community
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
        
        delta_modularity = get_delta_Q(C_A, s, C_norm)
        
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
        The modularity matrix for 'data_set'

    Returns
    -------
    C_A: np.matrix
        The modularity of 'comm_verts' within 'data_set'
    '''
    
    indices = tuple(u - 1 for u in comm_verts)
    C_A = B[indices, :][:, indices]
    
    return C_A

def get_delta_Q(C_A, s, C_norm):
    '''
    Calculates the modularity change according to Eq. A16 in the reference paper

    Parameters
    ----------
    C_A: np.matrix
        The generalized modularity matrix
    s: np.array
        The membership vector
    C_norm: float
        The the sum of all the elements of the correlation matrix

    Returns
    -------
    delta_Q: float
        The corresponding modularity change
    '''
    
    C_AA = np.sum(C_A)
    delta_Q = ((s.T.dot(C_A)).dot(s) - C_AA)/(2.*C_norm)
    
    return delta_Q

def get_modularity(B, comm_dict, n_verts, N_set, C_norm):
    '''
    Calculates the modularity according to Eq. 36 in the reference paper

    Parameters
    ----------
    B: np.matrix
        The modularity matrix for 'data_set'
    comm_dict: dict
        A dictionary to store the membership of each vertex in 'data_set'
    n_verts: np.array
        np.arange(1, N_set + 1)
    N_set: int
        The number of vertices in 'data_set'
    C_norm: float
        The the sum of all the elements of the correlation matrix
    
    Returns
    -------
    Q: float
        The modularity of 'data_set' given 'comm_dict'
    '''
    
    foo = np.sum([B[i, j] for i, j in product(range(N_set), range(N_set)) \
                if comm_dict[n_verts[i]] == comm_dict[n_verts[j]]])
    Q = foo/C_norm
    
    return Q

def visual_data(data, lvl, lvl_comm_dict, names_list):
    '''
    Using the membership dictionary at the given level, orders 'data' and saves 
    the heatmap showing the values of the entries of the ordered correlation matrix

    Parameters
    ----------
    data: np.matrix
        The complete matrix of the time series
    lvl: int
        The number that identifies the iteration of the optimization process
    lvl_comm_dict: dict
        A dictionary to store the membership of each vertex at the given level
    names_list: list
        The names of the vertices
    '''
    
    N = len(data[0])
    
    lvl_comm_labels = set(lvl_comm_dict.values())
    lvl_ord_names = []
    lvl_indices = []
    for i in lvl_comm_labels:
        lvl_comm_names = list(u for u in lvl_comm_dict if lvl_comm_dict[u] == i)
        lvl_ord_names.extend(lvl_comm_names)
        for j in lvl_comm_names:
            for k, n in enumerate(names_list):
                if(j == n): lvl_indices.append(k)
    lvl_ord_data = data[:, lvl_indices]
    
    ord_C = np.corrcoef(lvl_ord_data, rowvar = False)
    
    fig, ax = plt.subplots()
    im = ax.imshow(ord_C, cmap = 'RdBu_r')
    im.set_clim(-1, 1)
    ax.figure.colorbar(im, ax = ax)
    '''
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(lvl_ord_names)
    ax.set_yticklabels(lvl_ord_names)
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")
    '''
    plt.savefig("lvl_{}.png".format(lvl), dpi = 1000)
    plt.close()