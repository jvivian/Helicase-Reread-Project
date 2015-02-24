#!usr/bin/env python2.7
# John Vivian

'''
1. Find .JSON Events
2. At adequate C-Score ( 0.x ), collect events and store sequences (per context)
3. Train HMMs separately 
'''

import sys, os, random, argparse
import numpy as np
import Methods
from PyPore.DataTypes import *

sys.path.append( '../Models' )
from Simple_Model import *

'''
## 1. Randomize 230 events into groups
# Find JSON Events 
source = '../Data/JSON'
for root, dirnames, filenames in os.walk(source):
    events = filenames
    
# Randomize List
random.shuffle( events )

# Break into equal groups
event_groups = [ events[i::3] for i in xrange(3) ]
for i in event_groups:
	print len(i)
'''

## Call in Frozen Set of JSONs to train on
source = '../Data/JSON/C_train/'
for root, dirnames, filenames in os.walk(source):
    events = filenames

## Build profiles for HMMs
print '\n-=Building Profile=-'
distributions, fourmers, C_profile, mC_profile, hmC_profile = build_profile()

## Create HMMs 
print '-=Creating Untrained HMM=-'
with open ( '../Data/HMMs/untrained.txt', 'r' ) as file:
	model = Model.read( file ) 
	#print '\nTraining HMM: Witholding group {}. Training size {}. Cscore: {}'.format( i+1, len(training), cscore )

indices = { state.name: i for i, state in enumerate( model.states ) }

#cscore = 0.9 ## This value was used so that each context had a training set of ~50 events.

print 'Iterating through Training Set: {}'.format( 3 )
## Create lists for the sequences to be used in training
C_tset, mC_tset, hmC_tset = [], [], [] 

for event_name in events:

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
	'''
	if max( [ x[0] for x in contexts ] ) >= cscore and max( [ x[0] for x in labels ] ) >= cscore:
		
		# Filter by min value
		contexts = [ x for x in contexts if x[0] >= 0.1 ]     
		labels = [ x for x in labels if x[0] >= 0.1 ]

		# Use Independent consensus to determine vector
		ichunk, icall = Methods.ind_consensus( contexts, labels, cscore )

		# Use label hard call to sort event
		if icall[1] == 'C':
			C_tset.append( means )
		elif icall[1] == 'mC':
			mC_tset.append( means )
		elif icall[1] == 'hmC':
			hmC_tset.append( means )

		print event_name, icall[1]
	'''
	C_tset.append( means )

'''
print 'Creating UBER MODEL (forks)'
with open( '../Data/HMMs/Complete_Model_Untrained.txt', 'w' ) as file:
    model.write( file )
model.train( C_tset )
with open( '../Data/HMMs/Complete_Model_Trained.txt', 'w' ) as file:
    model.write( file )
'''
print '-=Creating C HMM=-'
C_model = Hel308_model( C_profile, 'C-31', fourmers )

print '-=Creating mC HMM=-'
mC_model = Hel308_model( mC_profile, 'mC-31', fourmers)

print '-=Creating hmC HMM=-'
hmC_model = Hel308_model( hmC_profile, 'hmC-31', fourmers )

print '\nTraining Cytosine HMM'
with open( '../Data/HMMs/C-untrained.txt', 'w' ) as file:
    C_model.write( file )
C_model.train( C_tset )
with open( '../Data/HMMs/C-trained.txt', 'w' ) as file:
    C_model.write( file )
'''
print 'Training mC HMM'
with open( '../Data/HMMs/mC-untrained.txt', 'w' ) as file:
    mC_model.write( file )
mC_model.train( mC_tset )
with open( '../Data/HMMs/mC-trained.txt', 'w' ) as file:
    mC_model.write( file )

print 'Training hmC HMM'
with open( '../Data/HMMs/hmC-untrained.txt', 'w' ) as file:
    hmC_model.write( file )
hmC_model.train( hmC_tset )
with open( '../Data/HMMs/hmC-trained.txt', 'w' ) as file:
    hmC_model.write( file )
'''