#!usr/bin/env python2.7
# John Vivian

import sys, os, random

sys.path.append( '../Models' )

from Simple_Model import *


## Find .JSON Events
source = '../Data/JSON'
for root, dirnames, filenames in os.walk(source):
    files = filenames

#events = [ Event.from_json( '../Data/JSON/' + x ) for x in files ]

sequences = []

for file in files:
    event = Event.from_json( '../Data/JSON/' + file )
    means = [seg['mean'] for seg in event.segments]
    sequences.append( means )

print '\n-=Building Profile=-'
profile = build_profile()

print '-=Building Complete HMM=-'
model = Hel308_model( profile[0], 'PW-31', profile[1] )

print 'HMM before training'
print model
with open ( '../Data/Trained_HMM/untrained.txt', 'w' ) as file:
    model.write( file )

print 'Training HMM'
model.train( sequences )

print "HMM after training:"
print model

with open ( '../Data/Trained_HMM/trained.txt', 'w' ) as file:
    model.write( file )

'''
This is how to SAVE the HMM after you've created/trained it
with open( 'test.txt', 'w' ) as file:
    model.write( file )
'''