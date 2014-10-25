#!usr/bin/env python2.7
# 9-9-14
# John Vivian

'''
This program is designed to help answer question of dependence when rereading
single molecules.
'''

import numpy as np
import random

'''
## Import of Model, build_profile, parse_abf, and analyze_event
import sys, argparse

parser = argparse.ArgumentParser(description='Can run either simple or substep model')
parser.add_argument('-s','--substep', action='store_true', help='Imports substep model instead of simple')
args = vars(parser.parse_args())

sys.path.append( '../Models' )
if args['substep']:
    print '\n-=SUBSTEP=-'
    from Substep_Model import *
else:
    print '\n-=SIMPLE=-'
    from Simple_Model import *
'''

## Methods

# Single Read Methods

def first_chunk( contexts, labels, cscore=0.9 ):
    ''' use the first chunk of each group'''
    
    ## Pull out high quality events
    C = [ x for x in contexts if x[0] >= cscore ]     
    L = [ x for x in labels if x[0] >= cscore ]
    
    ## Select First Chunks
    C = C[0]
    L = L[0]
    
    ## Retrieve the chunk vector
    C = C[1]    # Use branch vector to compute score 
    L = L[1]    
    
    soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    
    hard_call = call( C, L ) 
    
    return soft_call, hard_call

def last_chunk( contexts, labels, cscore=0.9 ):
    
    ## Pull out high quality events
    C = [ x for x in contexts if x[0] >= cscore ]
    L = [ x for x in labels if x[0] >= cscore ]
    
    ## Select Last Chunks
    C = C[-1]
    L = L[-1]
    
    ## retrieve the chunk vector
    C = C[1]
    L = L[1]
    
    soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    
    hard_call = call( C, L )
    
    return soft_call, hard_call
    
def random_chunk( contexts, labels, cscore=0.9):
    
    ## Pull out high quality events
    C = [ x for x in contexts if x[0] >= cscore ]
    L = [ x for x in labels if x[0] >= cscore ]
    
    ## Select random Chunk
    C = random.choice(C)
    L = random.choice(L)
    
    ## Retrieve the chunk vector
    C = C[1]
    L = L[1]
    
    soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    
    hard_call = call( C, L )
    
    return soft_call, hard_call
    
# Multi Read Methods

def best_chunk( contexts, labels ):
    ''' best chunk '''
    
    ## Pull out best event and vector
    C = max( contexts, key=lambda x: x[0] )[1]
    L = max( labels, key=lambda x: x[0] )[1]
    
    ## Get softcall
    soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    
    cl_call = call( C, L )
    
    return soft_call, cl_call

def ind_consensus( contexts, labels, cscore=0.9):
    ''' independent consensus 
    (1 - Product(1-Cn)) for each branch.
    '''
    
    ## Pull out high quality events
    C = [ x for x in contexts if x[0] >= cscore ]     
    L = [ x for x in labels if x[0] >= cscore ] 
    
    C_prod, mC_prod, hmC_prod = 1, 1, 1
    
    for c in C:
        c = c[1]                # Retrieve Chunk Vector
        C_prod *= (1 - c[0])
        mC_prod *= (1 - c[1])
        hmC_prod *= (1 - c[2])
    
    C = [ 1-C_prod, 1-mC_prod, 1-hmC_prod ]
    
    C_prod, mC_prod, hmC_prod = 1, 1, 1
    
    for l in L:
        l = l[1]
        C_prod *= (1 - l[0])
        mC_prod *= (1 - l[1])
        hmC_prod *= (1 - l[2])
    
    L = [ 1-C_prod, 1-mC_prod, 1-hmC_prod ]
    
    soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    
    hard_call = call( C, L )
    
    return soft_call, hard_call

def hmm_consensus( indices, ems, obs_len, chunk_vector ):
    ''' full consensus '''
    
    pseudo_contexts = [ [ None, [x for x in xrange(obs_len) ] ] ]
    pseudo_labels = [ [ None, [x for x in xrange(obs_len) ] ] ]
    
    contexts, labels = chunk_vector( indices, pseudo_contexts, pseudo_labels, ems )
    
    C = contexts[0][1]
    L = labels[0][1]
    
    soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    
    hard_call = call( C, L )
    
    return soft_call, hard_call

## Hard Call
def call( C, L ):
    ''' produces a "call" for a given list based on max '''
    ## Get a cytosine call
    ind = C.index(max(C))
    if ind == 0:
        c_call = 'C'
    elif ind == 1:
        c_call = 'mC'
    else:   
        c_call = 'hmC'
    
    ind = L.index(max(L))
    if ind == 0:
        l_call = 'C'
    elif ind == 1:
        l_call = 'mC'
    else:
        l_call = 'hmC'
        
    return ( c_call, l_call )
    

#################################
#								#
#  		End of Functions		#
#								#
#################################
'''
print '\n-=Building Profile=-'
profile = build_profile()

print '-=Building Complete HMM=-'
model = Hel308_model( profile[0], 'PW-31', profile[1] )
indices = { state.name: i for i, state in enumerate( model.states ) }

counter = 0
bins = { 'f': 0, 'l': 0, 'r':0, 'b': 0, 'i': 0, 'h': 0 }
files = ['14716003-s01.abf', '14714002-s01.abf', '14710002-s01.abf', '14710001-s01.abf' ]

for file in files:
    print '\n-=Parsing ABF=-'
    print '\tFile: {}'.format( file )
    for event in parse_abf('../Data/Mixed/'+file):

        ## Convert the Event into a list of segment means
        means = [seg.mean for seg in event.segments]
       
        ## Perform forward_backward algorithm
        trans, ems = model.forward_backward( means )
       
        ## Analyze Event to get a Filter Score
        data = analyze_event( model, event, trans, output=False )
        fscore = data['Score']
        
        
        ## If event passes Event Filter Score
        if fscore > .5:
            
            ## Partition the event into 'chunks' of context / label regions
            contexts, labels = partition_event( indices, event, ems, means)
            
            ## Get chunk scores
            contexts, labels = chunk_score( indices, contexts, labels, ems )
            
            ## Get chunk vector
            contexts, labels = chunk_vector( indices, contexts, labels, ems )
            
            if max( [ x[0] for x in contexts ] ) >= 0.9 and max( [ x[0] for x in labels ] ) >= 0.9:
                counter += 1
                print '\nEvent #{} Fscore: {} \tat: {}'.format( counter, round(fscore,4) , round(event.start, 2) )
                ## Single Read Methods
                fchunk, fcall = first_chunk( contexts, labels )
                lchunk, lcall = last_chunk( contexts, labels )
                rchunk, rcall = random_chunk( contexts, labels )
                
                ## Multi-Read Methods
                bchunk, bcall = best_chunk( contexts, labels )
                ichunk, icall = ind_consensus( contexts, labels )
                hchunk, hcall = hmm_consensus( indices, ems, len(means) )
                
                #############
                #   Output  #
                #############
                
                print '-=Single Read Methods=-'
                print 'First Chunk: {:<11} Hard Call: {:<4}, Label: {}'.format( round(fchunk,4), fcall[0], fcall[1] )
                if fcall[0] == fcall[1]:
                    bins['f'] += 1
        
                print 'Last Chunk: {:<11} Hard Call: {:<4}, label: {}'.format( round(lchunk,4), lcall[0], lcall[1] )
                if lcall[0] == lcall[1]:
                    bins['l'] += 1
                    
                print 'Random Chunk: {:<11} Hard Call: {:<4}, Label: {}'.format( round(rchunk,4), rcall[0], rcall[1] )
                if rcall[0] == rcall[1]:
                    bins['r'] += 1
                
                print '-=Multi-Read Methods=-'
                print 'Best Chunk: {:<11} Hard Call: {:<4}, Label: {}'.format( round(bchunk,4), bcall[0], bcall[1] )
                if bcall[0] == bcall[1]:
                    bins['b'] += 1
                    
                print 'Ind Consensus: {:<11} Hard Call: {:<4}, Label: {}'.format( round(ichunk,4), icall[0], icall[1] ) 
                if icall[0] == icall[1]:
                    bins['i'] += 1
                    
                print 'HMM Consensus: {:<11} Hard Call: {:<4}, Label: {}'.format( round(hchunk,4), hcall[0], hcall[1] ) 
                if hcall[0] == hcall[1]:
                    bins['h'] += 1


print counter
print bins
'''