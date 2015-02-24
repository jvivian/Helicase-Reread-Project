#!usr/bin/env python2.7
# John Vivian
# 10-8-14

## Import of Model, build_profile, parse_abf, and analyze_event, and segment_ems_plot
import sys, argparse

parser = argparse.ArgumentParser(description='Can run either simple or substep model')
parser.add_argument('-s','--substep', action='store_true', help='Imports substep model')
parser.add_argument('-p', '--pseudo', action='store_true', help='Imports pseudo-trained model')
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
distributions, fourmers, C, mC, hmC = build_profile()

print '-=Building HMM=-'
if args['substep']:
    model = Hel308_model( distributions, 'Test-43', fourmers )
elif args['pseudo']:
    print 'Pseudo Model'
    with open ( '../Data/HMMs/profile_trained_no_hmC.txt', 'r' ) as file:
        model = Model.read( file ) 
else:
    model = Hel308_model( distributions, 'Test-31', fourmers )

print '-=Parsing ABF=-'
for event in parse_abf('../Data/Training_Set/Mixed/14n13002-s04.abf', 544, 545):
    
    #print '-=Determining Viterbi Path=-'
    viterbi(model, event, fourmers)
    
    ## Perform the forward_backward algorithm to return transmission and emission matrices
    means = [seg.mean for seg in event.segments]
    trans, ems = model.forward_backward( means )
    
    ## Summary Information 
    data = analyze_event( model, event, trans )

    ## Plots
    segment_ems_plot( model, event, ems)   
    