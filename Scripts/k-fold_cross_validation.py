#!usr/bin/env python2.7
# John Vivian

'''
This script will perform k-fold cross-validation with our training set of JSON events (230). 


1. Randomize 230 events into 5 groups of 46 events
2. For every group: train on the other 4 sets
3. Read in HMM, Train
4. Test on the withheld group
3. Return Hard and Softcalls for each method:
    F_h, F_s, L_h, L_s, ... ( Method_hard/soft )
4. Store in array (5,12)
5. Output results.

6. Meta: Store each array in a dictionary / 3d array for different filter scores.
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


## 1. Randomize 230 events into 5 groups of 46 events
# Find JSON Events 
source = '../Data/JSON/FINAL_Train/'
for root, dirnames, filenames in os.walk(source):
    events = filenames
    
# Randomize List
random.shuffle( events )

# Break into 5 equal groups
event_groups = [ events[i::5] for i in xrange(5) ]
    
## 2. For every group: withold and train on other 4. 
# Create array
data = np.zeros( (1, 12) ) 

## 3. Read in Untrained HMM then train
with open ( '../Data/HMMs/Temp_Test.txt', 'r' ) as file:
    model = Model.read( file ) 
#print '\nTraining HMM: Witholding group {}. Training size {}. Cscore: {}'.format( i+1, len(training), cscore )
#model.train( sequences )

# Acquire indices
indices = { state.name: i for i, state in enumerate( model.states ) }

## In order to speed this thing up, assign events to lists by cscore so that
## it doesn't need to be done every iteration.
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
            if max_c >= i*.10 and max_l >= i*.10:
                print 'C:{}\tL:{}\tAssigned:{}\tPercentage:{}%\r'.format(round(max_c,2), round(max_l,2), i, round((counter*1.0/len(events))*100,2)),
                ranked_events[i].append( (event_name, contexts, labels, ems, means) )
                break
print '\n'
for i in ranked_events:
    print i, len(ranked_events[i])

## Iterate through the range of cutoff values: 
cscores = [ 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0 ]

for cscore in cscores:
    ## Keep track of hard calls -- list with 'n' and 'correct'
    hard_calls = { 'C': [0, 0], 'mC' : [0,0], 'hmC' : [0,0] }

    counters = []
    #for i in xrange(1): # 5 for real k-fold
    counter = 0

    # Bins to hold counts
    bins = { 'f': 0, 'l': 0, 'r':0, 'b': 0, 'i': 0, 'h': 0 }                # Counter for hard calls
    soft_calls = { 'f': [], 'l': [], 'r':[], 'b': [], 'i': [], 'h': [] }    # Will hold soft calls
    
    ## For a given cscore, group and iterate through.
    event_sum = 0
    for i in xrange(9, int(cscore*10-1), -1):
        event_sum += len(ranked_events[i])
        for event in ranked_events[i]:
            # Unpack Variables
            event_name = event[0]
            contexts = event[1]
            labels = event[2]
            ems = event[3]
            means = event[4]
            barcode = event_name.split('-')[0]
            # Counter for keeping track of number of events
            counter += 1
            ## Single Read Methods
            fchunk, fcall = Methods.first_chunk( contexts, labels, cscore )
            lchunk, lcall = Methods.last_chunk( contexts, labels, cscore )
            rchunk, rcall = Methods.random_chunk( contexts, labels, cscore )
            
            ## Multi-Read Methods
            bchunk, bcall = Methods.best_chunk( contexts, labels )
            ichunk, icall = Methods.ind_consensus( contexts, labels, cscore )
            hchunk, hcall = Methods.hmm_consensus( indices, ems, len(means), chunk_vector )

            #-=Single Read Methods=-
            # First Chunk
            soft_calls['f'].append( fchunk )
            if fcall[0] == barcode:
                bins['f'] += 1

            # Last Chunk
            soft_calls['l'].append( lchunk )
            if lcall[0] == barcode:
                bins['l'] += 1
                
            # Random Chunk
            soft_calls['r'].append( rchunk )
            if rcall[0] == barcode:
                bins['r'] += 1
            
            #-=Multi-Read Methods=-
            # Best Chunk
            soft_calls['b'].append( bchunk )
            if bcall[0] == barcode:
                bins['b'] += 1
                
            #Ind Consensus
            soft_calls['i'].append( ichunk )
            
            ## Increment count for hard calls
            hard_calls[ icall[1] ][0] += 1  # increment 'n' count
            
            if icall[0] == barcode:
                bins['i'] += 1

                ## Increment count for hard calls
                hard_calls[ icall[0] ][1] += 1 # Increment correct count
            print 'Context_hc: {}\tBarcode: {}\tEvent_Name:{}'.format( icall[0], barcode, event_name )
             
            #HMM Consensus
            soft_calls['h'].append( hchunk )
            if hcall[0] == barcode:
                bins['h'] += 1
            
            #print event_name, 'Label: ', icall[1], 'Context: ', icall[0]
            ## Add results to array
            if counter > 0:
                j = 0
                data[j][0] = bins['f']*1.0 / counter
                data[j][1] = np.mean(soft_calls['f'])
                data[j][2] = bins['l']*1.0 / counter
                data[j][3] = np.mean(soft_calls['l'])
                data[j][4] = bins['r']*1.0 / counter
                data[j][5] = np.mean(soft_calls['r'])
                data[j][6] = bins['h']*1.0 / counter
                data[j][7] = np.mean(soft_calls['h'])
                data[j][8] = bins['b']*1.0 / counter
                data[j][9] = np.mean(soft_calls['b'])
                data[j][10] = bins['i']*1.0 / counter
                data[j][11] = np.mean(soft_calls['i'])
            
            #print counter
            #print bins
            counters.append( counter )
    
    #print '\n', data, '\nSample Size: ~{}'.format( np.mean(counters) )

    #np.savetxt( '../Data/Results/Trained_All_' + str(event_sum) \
    #            + '_cscore_'+ str(cscore).split('.')[1] + '.txt', data, delimiter = ',' )
    print '\nCscore: {}'.format( cscore )
    for i in hard_calls:
        print i, 'n: {}, Correct: {}'.format( hard_calls[i][0], hard_calls[i][1] ) 

    print data


