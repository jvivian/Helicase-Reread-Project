#!/usr/bin/env python2.7
# John Vivian
# 1-25-15

'''
Output events to file
'''

import sys, os, random, argparse
import numpy as np
import Methods

sys.path.append( '../Models' )
from Simple_Model import *

# Find JSON Events 
source = '../Data/JSON/FINAL_Test/'
for root, dirnames, filenames in os.walk(source):
    events = filenames
    
# Read in model    
with open ( '../Data/HMMs/Frozen_HMM.txt', 'r' ) as file:
    model = Model.read( file ) 
    
# Acquire indices
indices = { state.name: i for i, state in enumerate( model.states ) }

# Rank Events by CHUNK Score
ranked_events = {}
for i in xrange(10):
    ranked_events[i] = []
    
print 'Ranking Events by CHUNK Score'
counter = 0
for event_name in events:  
    counter+=1
    # Convert JSON to event
    event = Event.from_json( source + event_name )

    # Convert event into a list of means
    means = [seg['mean'] for seg in event.segments]

    # Perform forward_backward algorithm
    trans, ems = model.forward_backward( means )

    # Partition the event into 'chunks' of context / label regions
    contexts, labels = partition_event( indices, event, ems, means)

    # Get chunk scores
    contexts, labels = chunk_score( indices, contexts, labels, ems )

    # Get chunk vector
    contexts, labels = chunk_vector( indices, contexts, labels, ems )
    
    if contexts and labels:
        max_c = max( [ x[0] for x in contexts ] ) 
        max_l = max( [ x[0] for x in labels ] )
        
        for i in xrange(9,-1,-1):
            if max_c >= i*.10:# and max_l >= i*.10:

                C = [ x for x in contexts if x[0] >= i*.10 ]
                L = [ x for x in labels if x[0] >= i*.10 ]
                
                sys.stdout.write( 'C:{}\t\tAssigned:{}\tPercentage:{}%\r'.format(round(max_c,2), i, round((counter*1.0/len(events))*100,2)))
                sys.stdout.flush()
                
                if len(C) > 1:
                    multi=True
                else:
                    multi=False
                C = sorted(C, key=lambda x: x[0], reverse=True)   
                ranked_events[i].append( (event_name, C, L, multi, ems, means) )
                
            

print '# of Events: {}'.format( len(ranked_events) )

print "Ranking Events"

ranked_events[0] = sorted( ranked_events[0], key=lambda x: x[1][0], reverse=True)

with open('..\Data\Ranked_Events\Events.txt', 'w') as file:
    for C in ranked_events[0]:
        file.write(str(C[0]) + "@" + str(C[1]) + "@" + str(C[2])+'\n')
    