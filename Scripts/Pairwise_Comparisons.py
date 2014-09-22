#!usr/bin/env python2.7
# 9-9-14
# John Vivian

'''
This program is designed to help answer question of dependence when rereading
single molecules.

1. A file will broken up into events by PyPore ( https://github.com/jmschrei/PyPore )
2. These events will be fed into an HMM ( https://github.com/jmschrei/yahmm )
3. 


'''

import pandas as pd
from yahmm import *
from PyPore.DataTypes import *

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
        if step_count in xrange( 21, 31 ):
            board.add_transition( match, match,         0.14 )
            board.add_transition( match, board.e2,      0.80 )
            board.add_transition( match, insert,        0.02 )
            board.add_transition( match, board.s3,      0.02 )
        else:
            board.add_transition( match, match,         0.16 )
            board.add_transition( match, board.e2,      0.80 )
            board.add_transition( match, insert,        0.02 )
            board.add_transition( match, board.s3,      0.02 )
    
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
        dists[name] = [ NormalDistribution( m, s ) for m, s in zip( means, stds )]

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

def analyze_event(model, event, trans):
    '''
    Uses the Forward-Backward Algorithm to determine expected transitions
    to certain states.
    '''
    
    indices = { state.name: i for i, state in enumerate( model.states ) }

    ## Pull out indices for the appropriate group into a list
    temp_C = [ x for x in indices.keys() if '(C)' in x and 'b' not in x and 'D' not in x and 'I' not in x ]
    temp_mC = [ x for x in indices.keys() if '(mC)' in x and 'b' not in x and 'D' not in x and 'I' not in x  ]
    temp_hmC = [ x for x in indices.keys() if '(hmC)' in x and 'b' not in x and 'D' not in x and 'I' not in x ]
    inserts = [ x for x in indices.keys() if 'I' in x ]
    deletes = [ x for x in indices.keys() if 'D' in x ]
    b1 = [ x for x in indices.keys() if 'e3' in x ]
    b2 = [ x for x in indices.keys() if 'e4' in x ]
    b3 = [ x for x in indices.keys() if 'e5' in x ]
    b4 = [ x for x in indices.keys() if 'e6' in x ]
    b5 = [ x for x in indices.keys() if 'e7' in x ]
    dis = [ x for x in indices.keys() if 'e8' in x ]
    
    ## Create lists to separate fork and tag indices
    C_fork = []; C_tag = []
    mC_fork = []; mC_tag = []
    hmC_fork = []; hmC_tag = []

    ## Parse 'temp' lists fork/tag lists
    for i in xrange(7,12):  # (6-10) = (7-11).  Changed to 8-10 given heatmap results.
        for match in temp_C:
            if ':'+str(i) in match:
                C_fork.append(match)
        for match in temp_mC:
            if ':'+str(i) in match:
                mC_fork.append(match)
        for match in temp_hmC:
            if ':'+str(i) in match:
                hmC_fork.append(match)

    for i in xrange(17,22):	# (16-20) = (17-21) because of I0 insert state.
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

    data[ 'C' ] = min( [ trans[ indices[name] ].sum() for name in C_fork ] )
    data[ 'mC' ] = min( [ trans[ indices[name] ].sum() for name in mC_fork ] )
    data[ 'hmC' ] = min( [ trans[ indices[name] ].sum() for name in hmC_fork ] )

    data[ 'mC-tag' ] = min( [ trans[ indices[name] ].sum() for name in mC_tag ] )
    data[ 'C-tag' ] = min( [ trans[ indices[name] ].sum() for name in C_tag ] )
    data[ 'hmC-tag' ] = min( [ trans[ indices[name] ].sum() for name in hmC_tag ] )

    ## Score
    data['Score'] = sum( data[tag] for tag in tags[:3] ) * sum( data[tag] for tag in tags[3:] )
    ## Soft Call
    score = data['C']*data['C-tag'] + data['mC']*data['mC-tag'] + data['hmC']*data['hmC-tag']
    data['Soft Call'] =  score / data['Score']

    ## Backslips ##
    b1_trans = [ float(trans[indices[x]].sum()) for x in b1 ]
    b2_trans = [ float(trans[indices[x]].sum()) for x in b2 ]
    b3_trans = [ float(trans[indices[x]].sum()) for x in b3 ]
    b4_trans = [ float(trans[indices[x]].sum()) for x in b4 ]
    b5_trans = [ float(trans[indices[x]].sum()) for x in b5 ]
    dis_trans = [ float(trans[indices[x]].sum()) for x in dis ] 

    ## Inserts / Deletes
    d_trans = [ float(trans[indices[x]].sum()) for x in inserts ]
    i_trans = [ float(trans[indices[x]].sum()) for x in deletes ]
    
    ## Reread
    r_trans = float ( trans[ indices [ 'reread' ] ].sum() )
    
    return data
    
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
L_model = C_L_Model( L[0], 'L-5', L[1] )

print '-=Parsing ABF=-'
for event in parse_abf('../Data/Mixed/14710002-s01.abf'):

    ## Convert the Event into a list of segment means
    means = [seg.mean for seg in event.segments]
    ## Perform forward_backward algorithm
    trans, ems = full_model.forward_backward( means )
    ## Analyze Event for Criteria (Filter Score and rereads)
    data = analyze_event( full_model, event, trans )

    #####################
    #	Event Filter	#
    #####################

    ## Unpack filter score and # of rereads
    fscore = round( data['Score'], 4 )
    
    if fscore > .50:
        print '\tSuitable event: {}'.format ( event.start )
        print '\tFilter Score: {}\n'.format( fscore )
        
