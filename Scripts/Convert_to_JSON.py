#!usr/bin/env python2.7
# John Vivian

import sys, os

sys.path.append( '../Models' )
from Simple_Model import *
import Methods

#####################
#   ABF Path Here   #
#####################
source = '../Data/Training_Set/Mixed' 
files = []
for root, dirnames, filenames in os.walk(source):
    files = filenames

print '\n-=Building Profile=-'
profile = build_profile()

print '-=Building HMM from Profile (Untrained)=-'
model = Hel308_model( profile[0], 'PW-31', profile[1] )
#with open ( '../Data/HMMs/profile_trained_no_hmC.txt', 'r' ) as file:
#    model = Model.read( file ) 
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
    
        ## Crude Filter
        if fscore > 0.5:

            segment_ems_plot( model, event, ems)    # Plot of Event
            choice = raw_input('\nC/M/H (or Enter to pass): ').upper()     # C/M/H or Enter for pass
            if choice == 'C':
                event.to_json ( '../Data/JSON/C-' + file.split('-')[0] + '-' + str(round(event.start,2)) +'.json' )
                print '\n\tC Event Added: {}'.format( file.split('-')[0] + '-' + str(round(event.start,2)) )
            elif choice == 'M':
                event.to_json ( '../Data/JSON/mC-' + file.split('-')[0] + '-' + str(round(event.start,2)) +'.json' )
                print '\n\tmC Event Added: {}'.format( file.split('-')[0] + '-' + str(round(event.start,2)) )
            elif choice == 'H':
                event.to_json ( '../Data/JSON/hmC-' + file.split('-')[0] + '-' + str(round(event.start,2)) +'.json' )
                print '\n\thmC Event Added: {}'.format( file.split('-')[0] + '-' + str(round(event.start,2)) )

print "You're done! Thanks Art -- You da real MVP"