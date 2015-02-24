#!usr/bin/env python2.7
# 9-9-14
# John Vivian

'''
Contains different methods for analyzing reads
'''

import numpy as np
import random


# Single Read Methods

def first_chunk( contexts, labels, cscore=0.9 ):
    ''' use the first chunk of each group'''
    
    ## Pull out high quality events
    C = [ x for x in contexts if x[0] >= cscore ]     
    #L = [ x for x in labels if x[0] >= cscore ]
    
    ## Select First Chunks
    C = C[0]
    #L = L[0]
    
    ## Retrieve the chunk vector
    C = C[1]    # Use branch vector to compute score 
    #L = L[1]    
    
    #soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    soft_call = 0
    
    hard_call = call( C )#, L ) 
    
    return soft_call, hard_call

def last_chunk( contexts, labels, cscore=0.9 ):
    
    ## Pull out high quality events
    C = [ x for x in contexts if x[0] >= cscore ]
    #L = [ x for x in labels if x[0] >= cscore ]
    
    ## Select Last Chunks
    C = C[-1]
    #L = L[-1]
    
    ## retrieve the chunk vector
    C = C[1]
    #L = L[1]
    
    #soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    soft_call = 0
    
    hard_call = call( C )#, L )
    
    return soft_call, hard_call

def random_chunk( contexts, labels, cscore=0.9):
    
    ## Pull out high quality events
    C = [ x for x in contexts if x[0] >= cscore ]
    #L = [ x for x in labels if x[0] >= cscore ]
    
    ## Select random Chunk
    C = random.choice(C)
    #L = random.choice(L)
    
    ## Retrieve the chunk vector
    C = C[1]
    #L = L[1]
    
    #soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    soft_call = 0
    
    hard_call = call( C )#, L )
    
    return soft_call, hard_call

    
# Multi Read Methods

def best_chunk( contexts, labels ):
    ''' best chunk '''
    
    ## Pull out best event and vector
    C = max( contexts, key=lambda x: x[0] )[1]
    #L = max( labels, key=lambda x: x[0] )[1]
    
    ## Get softcall
    #soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    soft_call = 0
    
    cl_call = call( C )#, L )
    
    return soft_call, cl_call

def ind_consensus( contexts, labels, cscore=0.9):
    ''' independent consensus 
    (1 - Product(1-Cn)) for each branch.
    '''
    
    ## Pull out high quality events
    C = [ x for x in contexts if x[0] >= cscore ]     
    #L = [ x for x in labels if x[0] >= cscore ] 
    
    C_prod, mC_prod, hmC_prod = 1, 1, 1
    
    for c in C:
        c = c[1]                # Retrieve Chunk Vector
        C_prod *= (1 - c[0])
        mC_prod *= (1 - c[1])
        hmC_prod *= (1 - c[2])
    
    C = [ 1-C_prod, 1-mC_prod, 1-hmC_prod ]
    # Normalize to sum to 1
    C = [ c/sum(C) for c in C ]
    
    
    '''
    for l in L:
        l = l[1]
        C_prod *= (1 - l[0])
        mC_prod *= (1 - l[1])
        hmC_prod *= (1 - l[2])
    
    L = [ 1-C_prod, 1-mC_prod, 1-hmC_prod ]\
    # Normalize to sum to 1
    L = [ l/sum(L) for l in L ]
    '''
    #soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    soft_call = 0
    
    hard_call = call( C )#, L )
    
    return soft_call, hard_call
    
def hmm_consensus( indices, ems, obs_len, chunk_vector ):
    ''' full consensus '''
    
    pseudo_contexts = [ [ None, [x for x in xrange(obs_len) ] ] ]
    pseudo_labels = [ [ None, [x for x in xrange(obs_len) ] ] ]
    
    contexts, labels = chunk_vector( indices, pseudo_contexts, pseudo_labels, ems )
    
    #L = labels[0][1]
    C = contexts[0][1]
    
    #soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    soft_call = 0
    
    hard_call = call( C )#, L )
    
    return soft_call, hard_call


# Hard Call

def call( C ):#, L ):
    ''' produces a "call" for a given list based on max '''
    ## Get a cytosine call
    ind = C.index(max(C))
    if ind == 0:
        c_call = 'C'
    elif ind == 1:
        c_call = 'mC'
    else:   
        c_call = 'hmC'
    
    '''
    ind = L.index(max(L))
    if ind == 0:
        l_call = 'C'
    elif ind == 1:
        l_call = 'mC'
    else:
        l_call = 'hmC'
    '''   
    return c_call # )
    
