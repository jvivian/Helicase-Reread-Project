#!usr/bin/env python2.7
# 9-9-14
# John Vivian

'''
This program is designed to help answer question of dependence when rereading
single molecules.
'''

import pandas as pd
from yahmm import *
from PyPore.DataTypes import *
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict

def Hel308_simple_model( distributions, name, fourmers, low=0, high=90 ):
    '''
    'Simple' Hel308 Model
    '''
    def BakeModule( distribution, step_count, fourmers, i=None, low=low, high=high ):
        '''
        The main "board" that comprises the HMM
        '''

        board = HMMBoard ( 8, name=str(i) )

        idx = str(i) if i else ""

        # Create the four states in the module
        insert = State( UniformDistribution( low, high ), name = 'I-{}'.format( idx ) )
        match = State( distribution, name=fourmers[step_count]+"-{}".format( idx ) )
        delete = State( None, name="D-{}".format( idx ) )
        
        ## Transitions
        # S1
        board.add_transition( board.s1, delete,     0.01 )
        board.add_transition( board.s1, match,      0.99 )
        # S2
        board.add_transition( board.s2, delete,     0.01 )
        board.add_transition( board.s2, match,      0.99 )
        ## Backslip / Dissocation
        # E3
        board.add_transition( board.e3, board.s4,   0.746 )
        board.add_transition( board.e3, match,      0.254 )
        # E4
        board.add_transition( board.e4, board.s5,   0.830 )
        board.add_transition( board.e4, match,      0.170 )
        # E5
        board.add_transition( board.e5, board.s6,   0.897 )
        board.add_transition( board.e5, match,      0.103 )
        # E6
        board.add_transition( board.e6, board.s7,   0.943 )
        board.add_transition( board.e6, match,      0.057 )
        # E7
        board.add_transition( board.e7, board.s8,   0.970 )
        board.add_transition( board.e7, match,      0.030 )
        # E8
        board.add_transition( board.e8, board.s8,   0.500 )
        board.add_transition( board.e8, match,      0.500 )
           
        ## Delete
        board.add_transition( delete, board.e1,     0.90 )
        board.add_transition( delete, insert,       0.10 )
        ## Insert
        board.add_transition( insert, match,        0.10 )
        board.add_transition( insert, insert,       0.50 )
        board.add_transition( insert, board.e2,     0.40 )
        ## Match
        if step_count in xrange( 21, 31 ):
            board.add_transition( match, match,         0.24 )
            board.add_transition( match, board.e2,      0.70 )
            board.add_transition( match, insert,        0.02 )
            board.add_transition( match, board.s3,      0.02 )
        else:
            board.add_transition( match, match,         0.25 )
            board.add_transition( match, board.e2,      0.70 )
            board.add_transition( match, insert,        0.04 )
            board.add_transition( match, board.s3,      0.01 )
    
        return board, match
    
    model = Model( name )
    boards = []
    
    ## Reread State
    reread = State (None, name='reread')
    
    for i, distribution in enumerate( distributions ):
        
        step_count = i
        
        if i > 0:
            pboard = boards[-1] # Prior Board
    
        if isinstance( distribution, Distribution ):
        
            board, match = BakeModule( distribution, step_count, fourmers, ":{}".format(i+1), low=low, high=high )
            model.add_model( board )

            ## Reread Contigency ################################
            if step_count in [0,5]:                             #
                model.add_transition (reread, match, 0.4)       #
            if step_count in [2,3]:                             #
                model.add_transition (reread, match, 0.07)      #
            if step_count in [4]:                               #
                model.add_transition (reread, match, 0.06)      #
            if step_count in xrange(21,31):                     #
                model.add_transition (match, reread, 0.02 )     #
            ## Reread Contigency ################################
            
            if i == 0:
                boards.append ( board )
                continue
            
            if isinstance( distributions[i-1], Distribution ):
                boards.append( board )
                
                model.add_transition( pboard.e1, board.s1,  1.00 )
                model.add_transition( pboard.e2, board.s2,  1.00 )
                model.add_transition( board.s3, pboard.e3,  1.00 )
                model.add_transition( board.s4, pboard.e4,  1.00 )
                model.add_transition( board.s5, pboard.e5,  1.00 )                
                model.add_transition( board.s6, pboard.e6,  1.00 )
                model.add_transition( board.s7, pboard.e7,  1.00 )
                model.add_transition( board.s8, pboard.e8,  1.00 )
                
            elif isinstance( distributions[i-1], dict ):
                n = len( distributions[i-1].keys() )

                for pboard in boards[-2*n+1::2]:
                    key = pboard.name.split()[1].split(':')[0].replace('(','').replace(')','')
                    
                    model.add_transition( pboard.e1, board.s1,  1.00 )
                    model.add_transition( pboard.e2, board.s2,  1.00 )
                    model.add_transition( board.s3, pboard.e3,  1.00 )
                    model.add_transition( board.s4, pboard.e4,  1.00 )
                    model.add_transition( board.s5, pboard.e5,  1.00 )                
                    model.add_transition( board.s6, pboard.e6,  1.00 )
                    model.add_transition( board.s7, pboard.e7,  1.00 )
                    model.add_transition( board.s8, pboard.e8,  1.00 )
                    
                boards.append( board )
            
        elif isinstance( distribution, dict ):
            n = len( distribution.keys() )
            for j, (key, dist) in enumerate( distribution.items() ):
            
                board, match = BakeModule( dist, step_count, fourmers, "({}):{}".format( key, i+1 ), low=low, high=high )
                model.add_model( board )
                boards.append ( board )
                
                ## Reread Contigency ################################
                if step_count in [0,5]:                             #
                    model.add_transition (reread, match, 0.4)       #
                if step_count in [2,3]:                             #
                    model.add_transition (reread, match, 0.07)      #
                if step_count in [4]:                               #
                    model.add_transition (reread, match, 0.06)      #
                if step_count in xrange(21,31):                     #
                    model.add_transition (match, reread, 0.02 )     #
                ## Reread Contigency ################################
            
                if isinstance( distributions[i-1], dict ):
                    boards.append( board )
                    pboard = boards[-2*n-1]
                    
                    model.add_transition( pboard.e1, board.s1,  1.00 )
                    model.add_transition( pboard.e2, board.s2,  1.00 )
                    model.add_transition( board.s3, pboard.e3,  1.00 )
                    model.add_transition( board.s4, pboard.e4,  1.00 )
                    model.add_transition( board.s5, pboard.e5,  1.00 )                
                    model.add_transition( board.s6, pboard.e6,  1.00 )
                    model.add_transition( board.s7, pboard.e7,  1.00 )
                    model.add_transition( board.s8, pboard.e8,  1.00 )
                    
                else:
                    boards.append( board )
                    
                    model.add_transition( pboard.e1, board.s1,  1.00 )
                    model.add_transition( pboard.e2, board.s2,  1.00 )
                    model.add_transition( board.s3, pboard.e3,  1.00 )
                    model.add_transition( board.s4, pboard.e4,  1.00 )
                    model.add_transition( board.s5, pboard.e5,  1.00 )                
                    model.add_transition( board.s6, pboard.e6,  1.00 )
                    model.add_transition( board.s7, pboard.e7,  1.00 )
                    model.add_transition( board.s8, pboard.e8,  1.00 )

    board = boards[0]
    initial_insert = State( UniformDistribution( low, high ), name="I:0" )
    model.add_state( initial_insert )

    model.add_transition( initial_insert, initial_insert, 0.70 )
    model.add_transition( initial_insert, board.s1, 0.1 )
    model.add_transition( initial_insert, board.s2, 0.2 )

    model.add_transition( model.start, initial_insert, 0.08 )
    model.add_transition( model.start, board.s1, 0.02 )
    model.add_transition( model.start, board.s2, 0.90 )

    ## Handling Reread and End Transitions
    board = boards[-1]
    # Transition to end
    model.add_transition( board.e1, model.end, .67 )
    model.add_transition( board.e2, model.end, .67 )
    # Transition to reread
    model.add_transition( board.e1, reread, .33 )
    model.add_transition( board.e2, reread, .33 )

    model.bake()
    return model
 
def C_L_Model( distributions, name, fourmers, low=0, high=90 ):
    '''
    Model for select Forks in the model
    '''
    def BakeModule( distribution, step_count, fourmers, i=None, low=low, high=high ):
        '''
        The main "board" that comprises the HMM
        '''

        board = HMMBoard ( 8, name=str(i) )

        idx = str(i) if i else ""

        # Create the four states in the module
        insert = State( UniformDistribution( low, high ), name = 'I-{}'.format( idx ) )
        match = State( distribution, name=fourmers[step_count]+"-{}".format( idx ) )
        delete = State( None, name="D-{}".format( idx ) )
        
        ## Transitions
        # S1
        board.add_transition( board.s1, delete,     0.05 )
        board.add_transition( board.s1, match,      0.95 )
        # S2
        board.add_transition( board.s2, delete,     0.05 )
        board.add_transition( board.s2, match,      0.95 )
        ## Backslip / Dissocation
        # E3
        board.add_transition( board.e3, board.s4,   0.746 )
        board.add_transition( board.e3, match,      0.254 )
        # E4
        board.add_transition( board.e4, board.s5,   0.830 )
        board.add_transition( board.e4, match,      0.170 )
        # E5
        board.add_transition( board.e5, board.s6,   0.897 )
        board.add_transition( board.e5, match,      0.103 )
        # E6
        board.add_transition( board.e6, board.s7,   0.943 )
        board.add_transition( board.e6, match,      0.057 )
        # E7
        board.add_transition( board.e7, board.s8,   0.970 )
        board.add_transition( board.e7, match,      0.030 )
        # E8
        board.add_transition( board.e8, board.s8,   0.500 )
        board.add_transition( board.e8, match,      0.500 )
           
        ## Delete
        board.add_transition( delete, board.e1,     0.90 )
        board.add_transition( delete, insert,       0.10 )
        ## Insert
        board.add_transition( insert, match,        0.10 )
        board.add_transition( insert, insert,       0.50 )
        board.add_transition( insert, board.e2,     0.40 )
        ## Match
        board.add_transition( match, match,         0.16 )
        board.add_transition( match, board.e2,      0.80 )
        board.add_transition( match, insert,        0.02 )
        board.add_transition( match, board.s3,      0.02 )
    
        return board, match
    
    model = Model( name )
    boards = []
    
    for i, distribution in enumerate( distributions ):
        
        step_count = i
        
        if i > 0:
            pboard = boards[-1] # Prior Board
    
        if isinstance( distribution, Distribution ):
        
            board, match = BakeModule( distribution, step_count, fourmers, ":{}".format(i+1), low=low, high=high )
            model.add_model( board )

            if i == 0:
                boards.append ( board )
                continue
            
            if isinstance( distributions[i-1], Distribution ):
                boards.append( board )
                
                model.add_transition( pboard.e1, board.s1,  1.00 )
                model.add_transition( pboard.e2, board.s2,  1.00 )
                model.add_transition( board.s3, pboard.e3,  1.00 )
                model.add_transition( board.s4, pboard.e4,  1.00 )
                model.add_transition( board.s5, pboard.e5,  1.00 )                
                model.add_transition( board.s6, pboard.e6,  1.00 )
                model.add_transition( board.s7, pboard.e7,  1.00 )
                model.add_transition( board.s8, pboard.e8,  1.00 )
                
            elif isinstance( distributions[i-1], dict ):
                n = len( distributions[i-1].keys() )

                for pboard in boards[-2*n+1::2]:
                    key = pboard.name.split()[1].split(':')[0].replace('(','').replace(')','')
                    
                    model.add_transition( pboard.e1, board.s1,  1.00 )
                    model.add_transition( pboard.e2, board.s2,  1.00 )
                    model.add_transition( board.s3, pboard.e3,  1.00 )
                    model.add_transition( board.s4, pboard.e4,  1.00 )
                    model.add_transition( board.s5, pboard.e5,  1.00 )                
                    model.add_transition( board.s6, pboard.e6,  1.00 )
                    model.add_transition( board.s7, pboard.e7,  1.00 )
                    model.add_transition( board.s8, pboard.e8,  1.00 )
                    
                boards.append( board )
            
        elif isinstance( distribution, dict ):
            n = len( distribution.keys() )
            for j, (key, dist) in enumerate( distribution.items() ):
            
                board, match = BakeModule( dist, step_count, fourmers, "({}):{}".format( key, i+1 ), low=low, high=high )
                model.add_model( board )
                boards.append ( board )

                if isinstance( distributions[i-1], dict ):
                    boards.append( board )
                    pboard = boards[-2*n-1]
                    
                    model.add_transition( pboard.e1, board.s1,  1.00 )
                    model.add_transition( pboard.e2, board.s2,  1.00 )
                    model.add_transition( board.s3, pboard.e3,  1.00 )
                    model.add_transition( board.s4, pboard.e4,  1.00 )
                    model.add_transition( board.s5, pboard.e5,  1.00 )                
                    model.add_transition( board.s6, pboard.e6,  1.00 )
                    model.add_transition( board.s7, pboard.e7,  1.00 )
                    model.add_transition( board.s8, pboard.e8,  1.00 )
                    
                else:
                    boards.append( board )
                    
                    model.add_transition( pboard.e1, board.s1,  1.00 )
                    model.add_transition( pboard.e2, board.s2,  1.00 )
                    model.add_transition( board.s3, pboard.e3,  1.00 )
                    model.add_transition( board.s4, pboard.e4,  1.00 )
                    model.add_transition( board.s5, pboard.e5,  1.00 )                
                    model.add_transition( board.s6, pboard.e6,  1.00 )
                    model.add_transition( board.s7, pboard.e7,  1.00 )
                    model.add_transition( board.s8, pboard.e8,  1.00 )

    board = boards[0]
    initial_insert = State( UniformDistribution( low, high ), name="I:0" )
    model.add_state( initial_insert )

    model.add_transition( initial_insert, initial_insert, 0.70 )
    model.add_transition( initial_insert, board.s1, 0.1 )
    model.add_transition( initial_insert, board.s2, 0.2 )

    model.add_transition( model.start, initial_insert, 0.08 )
    model.add_transition( model.start, board.s1, 0.02 )
    model.add_transition( model.start, board.s2, 0.90 )

    ## Handling Reread and End Transitions
    board = boards[-1]
    # Transition to end
    model.add_transition( board.e1, model.end, 1.0 )
    model.add_transition( board.e2, model.end, 1.0 )

    model.bake()
    return model
 
def build_profile( ):
    '''
    Reads in an excel to obtain profile information.
    '''
    profile, dists = [], {}
    c_profile, l_profile = [], []
    data = pd.read_excel( '../Profile/CCGG.xlsx', 'Sheet3' )

    total_means, total_stds = data.mean(axis=0), data.std(axis=0) # Total Profile Man
    total = [ NormalDistribution( m, 1.5 ) for m, s in zip( total_means, total_stds) ]

    for name, frame in data.groupby('label'):
        means, stds = frame.mean(axis=0), frame.std(axis=0)
        dists[name] = [ NormalDistribution( m, s ) for m, s in zip( means[:11], stds[:11] )]
        dists[name].extend([ NormalDistribution( m, 1.5 ) for m, s in zip( means[11:], stds[11:] )])
        
    # Piecing the profile together (0-3) = states 1-4
    for i in xrange(6):
        profile.append(total[i])
    
    c_profile.append(total[5])
    # Cytosine fork (6-10) = states 7-11
    for i in xrange(6, 11):
        profile.append( { 'C': dists['C'][i], 'mC': dists['mC'][i], 'hmC': dists['hmC'][i] } )
        c_profile.append( { 'C': dists['C'][i], 'mC': dists['mC'][i], 'hmC': dists['hmC'][i] } )
        
    # Continuation of profile  (11-15) = states 12-16
    for i in xrange(11, 16):
        profile.append(total[i])
    
    l_profile.append(total[15])
    # Label Fork (16-20) = states 17-21
    for i in xrange(16,21):
        profile.append( { 'C': dists['C'][i], 'mC': dists['mC'][i], 'hmC': dists['hmC'][i] } )
        l_profile.append( { 'C': dists['C'][i], 'mC': dists['mC'][i], 'hmC': dists['hmC'][i] } )
        
    # Continuation of profile  (21-27) = states 22-28
    for i in xrange(21, 32):
        profile.append(total[i])
        
    fourmers = [ col.replace(' ', '_').replace('.1', '').replace('.2', '') for col in data ][1:] 
    fourmers = [a.encode('ascii', 'ignore') for a in fourmers]

    return (profile, fourmers), (c_profile, fourmers[5:11]), (l_profile, fourmers[15:21])

def parse_abf(abf, start=0, end=750):
    '''
    parses an abf file.
    '''
    means = []

    ## Parse File
    file = File(abf)
    file.parse(parser=lambda_event_parser(threshold=108, rules = [ lambda event: event.duration > 1, 
            lambda event: event.min > -500, lambda event: event.max < 110 ]) )	
    print '-=File Parsed=-'

    ## Crude Event Filter
    for event in file.events:
        if event.duration > 2 and start < event.start < end:
            event.filter( order=1, cutoff=2000 )
            event.parse( SpeedyStatSplit( prior_segments_per_second = 40, cutoff_freq=2000 ) )
            if len(event.segments) > 15:
                yield event

def analyze_event(model, event, trans, c_start=7, c_end=12, l_start=17, l_end=22):
    '''
    Uses the Forward-Backward Algorithm to determine expected transitions
    to certain states.
    '''
    
    indices = { state.name: i for i, state in enumerate( model.states ) }

    ## Pull out indices for the appropriate group into a list
    temp_C = [ x for x in indices.keys() if '(C)' in x and 'b' not in x and 'D' not in x and 'I' not in x ]
    temp_mC = [ x for x in indices.keys() if '(mC)' in x and 'b' not in x and 'D' not in x and 'I' not in x  ]
    temp_hmC = [ x for x in indices.keys() if '(hmC)' in x and 'b' not in x and 'D' not in x and 'I' not in x ]

    ## Create lists to separate fork and tag indices
    C_fork = []; C_tag = []
    mC_fork = []; mC_tag = []
    hmC_fork = []; hmC_tag = []

    ## Parse 'temp' lists fork/tag lists
    
    for i in xrange( c_start, c_end ):  # (6-10) = (7-11).  Changed to 8-10 given heatmap results.
        for match in temp_C:
            if ':'+str(i) in match:
                C_fork.append(match)
        for match in temp_mC:
            if ':'+str(i) in match:
                mC_fork.append(match)
        for match in temp_hmC:
            if ':'+str(i) in match:
                hmC_fork.append(match)
    
    for i in xrange( l_start, l_end ):	# (16-20) = (17-21) because of I0 insert state.
        for match in temp_C:
            if str(i) in match:
                C_tag.append(match)
        for match in temp_mC:
            if str(i) in match:
                mC_tag.append(match)
        for match in temp_hmC:
            if str(i) in match:
                hmC_tag.append(match)

    ## Create a dictionary that will hold the computed values
    data = {}
    data['C'] = []
    data['mC'] = []
    data['hmC'] = []
    data['C-tag'] = []
    data['mC-tag'] = []
    data['hmC-tag'] = []
    data['Score'] = []
    data['Soft Call'] = []
    tags = ['C', 'mC', 'hmC', 'C-tag', 'mC-tag', 'hmC-tag']
    
    try:
        data[ 'C' ] = min( [ trans[ indices[name] ].sum() for name in C_fork ] )
        data[ 'mC' ] = min( [ trans[ indices[name] ].sum() for name in mC_fork ] )
        data[ 'hmC' ] = min( [ trans[ indices[name] ].sum() for name in hmC_fork ] )
    except:
        data[ 'C' ] = (1.0/3)
        data[ 'mC' ] = (1.0/3)
        data[ 'hmC' ] = (1.0/3)
    try:
        data[ 'mC-tag' ] = min( [ trans[ indices[name] ].sum() for name in mC_tag ] )
        data[ 'C-tag' ] = min( [ trans[ indices[name] ].sum() for name in C_tag ] )
        data[ 'hmC-tag' ] = min( [ trans[ indices[name] ].sum() for name in hmC_tag ] )
    except:
        data[ 'mC-tag'] = (1.0/3)
        data[ 'C-tag' ] = (1.0/3)
        data[ 'hmC-tag' ] = (1.0/3)
        
    ## Score
    data['Score'] = sum( data[tag] for tag in tags[:3] ) * sum( data[tag] for tag in tags[3:] )
    ## Soft Call
    score = data['C']*data['C-tag'] + data['mC']*data['mC-tag'] + data['hmC']*data['hmC-tag']
    data['Soft Call'] =  score / data['Score']
    
    return data

def partition_event( model, event, ems, means ):
    '''
    Partitions event based on the emission matrix from the Forward-Backward Algorithm
    '''
    indices = { state.name: i for i, state in enumerate( model.states ) }

    ## Because the EMS table only contains emission states, then 'end' state
    ## associated with the match state cannot be used. Thus, the "Match" and
    ## "Match-Over" states will be summed.
    C_temp = [x for x in indices.keys() if '(C)' in x and 'I' not in x and 'b' not in x and 'D' not in x]
    mC_temp = [x for x in indices.keys() if '(mC)' in x and 'I' not in x  and 'b' not in x and 'D' not in x]
    hmC_temp = [x for x in indices.keys() if '(hmC)' in x and 'I' not in x  and 'b' not in x and 'D' not in x]	
    
    ## Separate into tags and forks for each context
    C_fork = [ x for x in C_temp if int(x.split(':')[1]) in xrange(7,12) ]
    C_tag = [ x for x in C_temp if int(x.split(':')[1]) in xrange(17, 22) ]
    
    mC_fork = [ x for x in mC_temp if int(x.split(':')[1]) in xrange(7,12) ]
    mC_tag = [ x for x in mC_temp if int(x.split(':')[1]) in xrange(17, 22) ]
    
    hmC_fork = [ x for x in hmC_temp if int(x.split(':')[1]) in xrange(7,12) ]
    hmC_tag = [ x for x in hmC_temp if int(x.split(':')[1]) in xrange(17, 22) ]
    
    all_contexts = C_fork + mC_fork + hmC_fork
    all_labels = C_tag + mC_tag + hmC_tag
    
    ## Get the index values for each name in the fork / label
    C_all = np.array( map( indices.__getitem__, all_contexts ) )
    L_all = np.array( map( indices.__getitem__, all_labels ) )
    c_dict, l_dict = OrderedDict(), OrderedDict()
    for i in xrange(7,12):
        c_dict[i] = np.array( map( indices.__getitem__, [x for x in all_contexts if ':'+str(i) in x] ) )
    for i in xrange(17, 22):
        l_dict[i] = np.array( map ( indices.__getitem__, [x for x in all_labels if ':'+str(i) in x] ) )
        
    ## Use those index values to compute probabilities by observation
    pC_all = np.exp( ems[ :, C_all ] ).sum( axis=1 )
    pL_all = np.exp( ems[:, L_all ] ).sum( axis=1 )
    
    ## Partition Events given emission matrix
    contexts, labels = [], []
    temp_c, temp_l = [], []
    
    for i in xrange(len(pC_all)): 
        if pC_all[i] > 0.5:         
            temp_c.append( (round(means[i],4), i) )
        else:
            if temp_c:
                temp = zip( *temp_c )
                contexts.append( [temp[0], temp[1]] )
                temp_c = []
    
    for i in xrange(len(pL_all)):
        if pL_all[i] > 0.5:
            temp_l.append( (round(means[i],4), i) )
        else:
            if temp_l:
                temp = zip( *temp_l )
                labels.append( [temp[0], temp[1]] )
                temp_l = []
                
    ## Obtain Prior for each Context ##
    p_dict = OrderedDict()
    pscore = []
    context_final = []
    weights = [ 1.0/9, 2.0/9, 1.0/3, 2.0/9, 1.0/9 ]
    for c in contexts:
        temp_ems = ems[ c[1], : ]                   # Slice matrix based on observations
        for i in xrange(7,12):
            p_dict[i] = np.max( np.exp( temp_ems[:, c_dict[i] ]).sum( axis=1 ) )
        
        ## Combine P_scores into a single score
        pscore = [ p_dict[x] for x in p_dict ] 
        pscore = [ a*b for a,b in izip(pscore, weights) ]
        pscore = sum(pscore)
        
        if pscore > 0.9:
            context_final.append( (round(pscore,4), c[0]) )
    
    ## Obtain Prior for each Label ##
    p_dict = OrderedDict()
    pscore = []
    label_final = []
    weights = [1.0/11, 2.0/11, 3.0/11, 3.0/11, 2.0/11]
    for l in labels:
        temp_ems = ems[ l[1], : ]
        for i in xrange( 17, 22):
            p_dict[i] = np.max( np.exp( temp_ems[:, l_dict[i] ]).sum( axis=1 ) )
        
        ## Combine P_scores into a single score
        pscore = [p_dict[x] for x in p_dict]
        pscore = [a*b for a,b in izip(pscore, weights) ]
        pscore = sum(pscore)
        
        if pscore > 0.9:
            label_final.append( (round(pscore,4), l[0]) )
            
    print context_final
    print
    print label_final
    
    return context_final, label_final
        
def scoring( contexts, labels, C_model, L_model ):
    '''
    '''
    filtered_c, filtered_l = [], []
    
    for i in xrange( len(contexts) ):
        trans, ems = C_model.forward_backward( contexts[i][0] )
        data = analyze_event( C_model, contexts[i], trans, 2, 7 )
        if data['Score'] * contexts[i][1] > .5:
            print '\tContext #{} \t Score: {}'.format( i+1, data['Score']*contexts[i][1] )
            filtered_c.append( data )
            
    for i in xrange( len(labels) ):
        trans, ems = L_model.forward_backward( labels[i][0] ) 
        data = analyze_event( L_model, labels[i], trans, 99, 99, 2, 7 )
        if data['Score'] * labels[i][1] > .5:
            print '\tLabel #{} \t Score: {}'.format( i+1,  data['Score']*labels[i][1] ) 
            filtered_l.append( data )
    print
    return filtered_c, filtered_l
    
#################################
#								#
#  		End of Functions		#
#								#
#################################

print '\n-=Building Profile=-'
full, C, L = build_profile()

print '-=Building Complete HMM=-'
full_model = Hel308_simple_model( full[0], 'PW-31', full[1] )

print '-=Building Context HMM=-'
C_model = C_L_Model( C[0], 'C-6', C[1] )

print '-=Building Label HMM=-'
L_model = C_L_Model( L[0], 'L-6', L[1] )

print '-=Parsing ABF=-'
master = []
counter = 0

file = '14710002-s01.abf'
print 'File: {}'.format( file )
for event in parse_abf('../Data/Mixed/'+file):
    ## Convert the Event into a list of segment means
    means = [seg.mean for seg in event.segments]
    ## Perform forward_backward algorithm
    trans, ems = full_model.forward_backward( means )
    ## Analyze Event for Criteria (Filter Score and rereads)
    data = analyze_event( full_model, event, trans )
    
    ## Unpack filter score 
    fscore = data['Score']
    counter += 1
    if fscore > .9:
        print 'Event #{} Fscore: {} \tat: {}'.format( counter, round(fscore,4) , round(event.start, 2) )
        contexts, labels = partition_event( full_model, event, ems, means)
        #C, L = scoring( contexts, labels, C_model, L_model )
        break
