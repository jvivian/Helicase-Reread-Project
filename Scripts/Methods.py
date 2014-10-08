#!usr/bin/env python2.7
# 9-9-14
# John Vivian

'''
This program is designed to help answer question of dependence when rereading
single molecules.
'''

import numpy as np
from collections import OrderedDict

## Import of Model, build_profile, parse_abf, and analyze_event
import sys
sys.path.append( '../Models' )
from Simple_Model import *


## Methods

def first_chunk( contexts, labels ):
    ''' use the first chunk of each group'''
    
    ## Pull out high quality events
    C = [ x for x in contexts if x[0] >= 0.9 ]     
    L = [x for x in labels if x[0] >= 0.9 ]
    
    ## Select first event
    C = C[0]
    L = L[0]
    
    ## Get a softcall
    C = C[1]    # Use branch vector to compute score 
    L = L[1]    
    
    soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    
    cl_call = call( C, L ) 
    
    return soft_call, cl_call

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
    
def best_chunk( contexts, labels ):
    ''' best chunk '''
    
    ## Pull out best event and vector
    C = max( contexts, key=lambda x: x[0] )[1]
    L = max( labels, key=lambda x: x[0] )[1]
    
    ## Get softcall
    soft_call = C[0]*L[0] + C[1]*L[1] + C[2]*L[2]
    
    cl_call = call( C, L )
    
    return soft_call, cl_call

def ind_consensus( contexts, labels):
    ''' independent consensus '''
    pass


#################################
#								#
#  		End of Functions		#
#								#
#################################

print '\n-=Building Profile=-'
profile = build_profile()

print '-=Building Complete HMM=-'
model = Hel308_model( profile[0], 'PW-31', profile[1] )
indices = { state.name: i for i, state in enumerate( model.states ) }

counter = 0
file = '14710002-s01.abf'
print '\tFile: {}'.format( file )
print '-=Parsing ABF=-'
for event in parse_abf('../Data/Mixed/'+file):

    ## Convert the Event into a list of segment means
    means = [seg.mean for seg in event.segments]
   
    ## Perform forward_backward algorithm
    trans, ems = model.forward_backward( means )
   
    ## Analyze Event to get a Filter Score
    data = analyze_event( model, event, trans, output=False )
    fscore = data['Score']
    counter += 1
    
    ## If event passes Event Filter Score
    if fscore > .9:
        print '\nEvent #{} Fscore: {} \tat: {}'.format( counter, round(fscore,4) , round(event.start, 2) )
        
        ## Partition the event into 'chunks' of context / label regions
        contexts, labels = partition_event( indices, event, ems, means)
        
        ## Get chunk scores
        contexts, labels = chunk_score( indices, contexts, labels, ems )
        
        ## Get chunk vector
        contexts, labels = chunk_vector( indices, contexts, labels, ems )
        
        if max( [ x[0] for x in contexts ] ) >= 0.9 and max( [ x[0] for x in labels ] ) >= 0.9:
            
            ## First Chunk
            sc_fchunk, fcall = first_chunk( contexts, labels )
            
            ## Best Chunk
            sc_bchunk, bcall = best_chunk( contexts, labels )
            
            print 'First Chunk: {}, Call: {}, Label: {}'.format( round(sc_fchunk,2), fcall[0], fcall[1] )
            print 'Best Chunk: {}, Call: {}, Label: {}'.format( round(sc_bchunk,2), bcall[0], bcall[1] )
