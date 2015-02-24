#!usr/bin/env python2.7
# John Vivian

'''
Test script for events
'''

import sys, os, random, argparse
import numpy as np
import Methods

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
    
################################

## Find JSON Events 
source = '../Data/JSON'
for root, dirnames, filenames in os.walk(source):
    events = filenames
    
## Read in HMM
if args['substep']:
    with open ( '../Data/HMMs/untrained_substep.txt', 'r' ) as file:
        model = Model.read( file )
else:
    with open ( '../Data/HMMs/untrained.txt', 'r' ) as file:
        model = Model.read( file )

cscore = 0.1  
indices = { state.name: i for i, state in enumerate( model.states ) }

for event in parse_abf('../Data/Training_set/14721001-s01.abf'):
    
    # Convert JSON to event
    #event = Event.from_json( '../Data/JSON/' + event_name )
    
    # Convert event into a list of means
    #means = [seg['mean'] for seg in event.segments]
    means = [seg.mean for seg in event.segments]
   
    # Perform forward_backward algorithm
    trans, ems = model.forward_backward( means )
   
    # Partition the event into 'chunks' of context / label regions
    contexts, labels = partition_event( indices, event, ems, means)

    # Get chunk scores
    contexts, labels = chunk_score( indices, contexts, labels, ems )
   
    # Get chunk vector
    contexts, labels = chunk_vector( indices, contexts, labels, ems )
    
    contexts = [ x for x in contexts if x[0] >= cscore ]     
    labels = [ x for x in labels if x[0] >= cscore ]
    if len(contexts) > 0 and len(labels) > 0:
    
        for i in contexts:
            print i[0], i[1]
        print
        for i in labels:
            print i[0], i[1]
        
        distributions, fourmers = build_profile()
        viterbi(model, event, fourmers)
        segment_ems_plot( model, event, ems)   
        
        
    