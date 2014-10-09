#!usr/bin/env python2.7
# John Vivian
# 10-8-14

## Import of Model, build_profile, parse_abf, and analyze_event, and segment_ems_plot
import sys, argparse

parser = argparse.ArgumentParser(description='Can run either simple or substep model')
parser.add_argument('-s','--substep', action='store_true', help='Imports substep model')
args = vars(parser.parse_args())

sys.path.append( '../Models' )
if args['substep']:
    print '\n-=SUBSTEP MODEL=-'
    from Substep_Model import *
else:
    print '\n-=SIMPLE Model=-'
    from Simple_Model import *



################################

print '\n-=Building Profile=-'
distributions, fourmers = build_profile()

print '-=Building HMM=-'
if args['substep']:
    model = Hel308_model( distributions, 'Test-43', fourmers )
else:
    model = Hel308_model( distributions, 'Test-31', fourmers )

print '-=Parsing ABF=-'
for event in parse_abf('../Data/Mixed/14710002-s01.abf', 259, 260):
    
    print '-=Determining Viterbi Path=-'
    viterbi(model, event, fourmers)
    
    ## Perform the forward_backward algorithm to return transmission and emission matrices
    means = [seg.mean for seg in event.segments]
    trans, ems = model.forward_backward( means )
    
    ## Summary Information 
    data = analyze_event( model, event, trans )

    ## Plots
    segment_ems_plot( model, event, ems)   
    
    '''
    This is how to SAVE the HMM after you've created/trained it
    with open( 'test.txt', 'w' ) as file:
        model.write( file )
    '''

