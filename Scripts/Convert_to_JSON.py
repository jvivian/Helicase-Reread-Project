#!usr/bin/env python2.7
# John Vivian

import sys, os

sys.path.append( '../Models' )
from Simple_Model import *

## Find .abfs
source = '../Data/Training_set'
files = []
for root, dirnames, filenames in os.walk(source):
    files = filenames

print files

print '\n-=Building Profile=-'
profile = build_profile()

print '-=Building Complete HMM=-'
model = Hel308_model( profile[0], 'PW-31', profile[1] )
indices = { state.name: i for i, state in enumerate( model.states ) }

for file in files:
    print '\n-=Parsing ABF=-'
    print '\tFile: {}'.format( file )
    for event in parse_abf(source+'//'+file):
        
        ## Convert the Event into a list of segment means
        means = [seg.mean for seg in event.segments]
       
        ## Perform forward_backward algorithm
        trans, ems = model.forward_backward( means )
       
        ## Analyze Event to get a Filter Score
        data = analyze_event( model, event, trans, output=False )
        fscore = data['Score']
        
        ## If event passes Event Filter Score
        if fscore > 0.5:
            
            file.to_json( '../Data/JSON/' + file.split('-')[0] + '.json' )
            
            '''
            ## Partition the event into 'chunks' of context / label regions
            contexts, labels = partition_event( indices, event, ems, means)
            
            ## Get chunk scores
            contexts, labels = chunk_score( indices, contexts, labels, ems )
            
            ## Get chunk vector
            contexts, labels = chunk_vector( indices, contexts, labels, ems )
            try:
                if max( [ x[0] for x in contexts ] ) >= 0 and max( [ x[0] for x in labels ] ) >= 0:
                    event.to_json ( '../Data/JSON/' + file.split('-')[0] + '-' + str(round(event.start,2)) +'.json' )
            except:
                pass
            '''