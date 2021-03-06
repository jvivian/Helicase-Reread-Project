#!usr/bin/env python2.7
# John Vivian

'''
Substep_Model.py is a collection of functions that perform the following operations:
    
    1.  Builds a profile from an external data file
    2.  Builds a Hidden Markov Model (HMM) from the profile data
    3.  Parses an .ABF file (nanopore data) in order to iterate through discrete events
    4.  Events are passed through the HMM to return a comprehensive Viterbi path.
    5.  Additional information culled from the Forward-Backward algorithm is also displayed
    6.  Plots the segmented event, the event as colored by HMM-board, and the emission probabilities 
        for the fork and label region
    
    For Substep Model:
        Context = States 10-17 [ C*GGT - ATCC ]
        Label = States 24-33  [ LTCA - CATL ]

'''

from yahmm import *
import pandas as pd
from PyPore.DataTypes import *
from collections import OrderedDict

def Hel308_model( distributions, name, fourmers, low=0, high=90 ):      ## Fixed for substeps
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
        if step_count in [3,6,8,13,15,20,24,26,28,30,32]:       # Altered Pr(Delete) for substeps
            # S1
            board.add_transition( board.s1, delete,     0.75 )
            board.add_transition( board.s1, match,      0.25 )
            # S2
            board.add_transition( board.s2, delete,     0.75 )
            board.add_transition( board.s2, match,      0.25 )
        else:
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
        if step_count in xrange( 31, 43 ):
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
            if step_count in [1,2,3,4,6]:                       #
                model.add_transition (reread, match, 0.04)      #    
            if step_count in xrange(31,43):                     #
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
                if step_count in [1,2,3,4,6]:                       #
                    model.add_transition (reread, match, 0.04)      #    
                if step_count in xrange(31,43):                     #
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

def build_profile( ):                                                   ## Fixed for substeps
    '''
    Reads in an excel to obtain profile for the HMM.  
    Profile is stored as a list of distributions with forks represented by dictionaries.
    '''
    profile, dists = [], {}
    data = pd.read_excel( '../Profile/CCGG.xlsx', 'SSp' )

    total_means, total_stds = data.mean(axis=0), data.std(axis=0) # Total Profile Man
    total = [ NormalDistribution( m, 1.5 ) for m, s in zip( total_means, total_stds) ]

    for name, frame in data.groupby('label'):
        means, stds = frame.mean(axis=0), frame.std(axis=0)
        dists[name] = [ NormalDistribution( m, s ) for m, s in zip( means[:17], stds[:17] )]
        dists[name].extend([ NormalDistribution( m, 1.5 ) for m, s in zip( means[17:], stds[17:] )])

    # Piecing the profile together 
    for i in xrange(9):
        profile.append(total[i])

    # Cytosine fork 9-16 : States 10-17
    for i in xrange(9, 17):
        profile.append( {'C': dists['C'][i], 'mC': dists['mC'][i], 'hmC': dists['hmC'][i]  })

    # Continuation of profile  
    for i in xrange(17, 23):
        profile.append(total[i])

    # Label Fork (23-31) = states 24-32
    for i in xrange(23,33):
        profile.append( {'C': dists['C'][i], 'mC': dists['mC'][i], 'hmC': dists['hmC'][i]  })

    # Continuation of profile  
    for i in xrange(33, 44):
        profile.append(total[i])
        
    fourmers = [ col.replace(' ', '_').replace('.1', '').replace('.2', '') for col in data ][1:] 
    fourmers = [a.encode('ascii', 'ignore') for a in fourmers]

    return profile, fourmers

def parse_abf(abf, start=0, end=750):                                   # No changes necessary
    '''
    parses an abf file and yields an event
    '''
    means = []

    ## Parse File
    file = File(abf)
    file.parse(parser=lambda_event_parser(threshold=108, rules = [ lambda event: event.duration > 1, 
            lambda event: event.min > -500, lambda event: event.max < 110 ]) )	
    print '\tFile: Parsed'

    ## Crude Event Filter
    for event in file.events:
        if event.duration > 2 and start < event.start < end:
            event.filter( order=1, cutoff=2000 )
            event.parse( SpeedyStatSplit( prior_segments_per_second = 40, cutoff_freq=2000 ) )
            if len(event.segments) > 15:
                yield event   
    
    file.delete()
    
def analyze_event(model, event, trans, output=True):                    ## Fixed for substeps
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
    for i in xrange(10,18):  # (6-10) = (7-11).  Changed to 8-10 given heatmap results.
        for match in temp_C:
            if ':'+str(i) in match:
                C_fork.append(match)
        for match in temp_mC:
            if ':'+str(i) in match:
                mC_fork.append(match)
        for match in temp_hmC:
            if ':'+str(i) in match:
                hmC_fork.append(match)

    for i in xrange(24,34):	# (16-20) = (17-21) because of I0 insert state.
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

    #############
    #	Output	#
    #############
    
    if output:
        print 'Soft Call:\t{}'.format( round( data['Soft Call'], 4 ) ) 
        print 'Filter Score:\t{}\n'.format( round ( data['Score'], 4 ) )
        print '-'*40, '\n'

        print 'C:\t\t{}'.format( round( data['C'], 4 ) )
        print 'mC:\t\t{}'.format( round( data['mC'], 4 ) )
        print 'hmC:\t\t{}\n'.format( round( data['hmC'], 4 ) )
        print 'C-tag:\t\t{}'.format( round( data['C-tag'], 4 ) )
        print 'mC-tag:\t\t{}'.format( round( data['mC-tag'], 4 ) )
        print 'hmC-tag:\t{}\n'.format( round( data['hmC-tag'], 4 ) )
        print '-'*40, '\n'

        print 'Estimated Number of Deletes: {}'.format( round( sum( d_trans), 1) ) 
        print 'Estimated Number of Inserts: {}'.format( round( sum( i_trans), 1) ) 
        print 'Estimated Number of Single Backslips: {}'.format( round( sum( b1_trans), 1 ) ) 
        print 'Estimated Number of Backslips (Length=2): {}'.format( round( sum( b2_trans), 1 ) )
        print 'Estimated Number of Backslips (Length=3): {}'.format( round( sum( b3_trans), 1 ) )
        print 'Estimated Number of Backslips (Length=4): {}'.format( round( sum( b4_trans), 1 ) )
        print 'Estimated Number of Backslips (Length=5): {}'.format( round( sum( b5_trans), 1 ) )
        print 'Estimated Number of Backslips (Min-length:6,+1 per additional): {}'.format( round( sum( dis_trans), 1) )
        print 'Estimated Number of Rereads: {}'.format ( round (r_trans), 1) 

    return data


## Viterbi Output and Plot
def viterbi(model, event, fourmers):                                    # No changes necessary
    '''
    Runs the Viterbi Algorithm -- produces a comprehensive output.
    '''

    ## Convert the Event into a list of segment means
    means = [seg.mean for seg in event.segments]
    ## Run that list through the viterbi algorithm and return emission states
    vit_path = [ state.name for i, state in model.viterbi(means)[1] if not state.is_silent() ]

    #############
    #	Output	#
    #############

    print '# of Observations (Segments): {}'.format( len(means) ) 
    print 'Start Time of Event: {}'.format( event.start )
    print '\n', '='*40
    print '\n{:^}\t\t{:<20} {:^5}'.format('Obs', 'Path', 'Mean'), '\n'

    counter = 0
    for i, state in enumerate(vit_path):
        if i < len(vit_path)-2:
            if state == vit_path[i+1]:
                counter += 1
            else:
                if counter == 0:
                    print '{:<}\t\t{:<20} {:<}'.format( i, state, round(means[i],1) )
                else:
                    mean_list = [ round(x, 1) for x in means[i-counter:i+1] ]
                    print '{}-{:<}\t\t{:<20} {:<}~'.format( i-counter, i, state, round(np.mean(mean_list),1) )
                    counter = 0
        else:
            if counter == 0:
                    print '{:<}\t\t{:<20} {:<}'.format( i, state, round(means[i],1) )
            else:
                mean_list = [ round(x, 1) for x in means[i-counter:i+1] ]
                print '{}-{:<}\t\t{:<20} {:<}~'.format( i-counter, i, state, round(np.mean(mean_list),1) )
                counter = 0

    print '\n', '='*40   

def segment_ems_plot( model, event, ems):                               ## Fixed for substeps
    '''
    Plots 3 items:  Segmented event, Event colored by HMM-state, and an emissions plot
    '''

    plt.subplot( 311 )
    event.plot( color='cycle')
    plt.ylim(-5, 100)

    plt.subplot( 312 )
    event.plot ( color='hmm', hmm=model )
    plt.ylim(-5, 100)

    indices = { state.name: i for i, state in enumerate( model.states ) }

    ## Because the EMS table only contains emission states, then 'end' state
    ## associated with the match state cannot be used. Thus, the "Match" and
    ## "Match-Over" states will be summed.
    C_temp = [x for x in indices.keys() if '(C)' in x and 'end' not in x and 'start' not in x and 'D' \
            not in x and 'I' not in x and 'b' not in x]
    mC_temp = [x for x in indices.keys() if '(mC)' in x and 'end' not in x and 'start' not in x and 'D' \
            not in x and 'I' not in x  and 'b' not in x]
    hmC_temp = [x for x in indices.keys() if '(hmC)' in x and 'end' not in x and 'start' not in x and 'D' \
            not in x and 'I' not in x  and 'b' not in x]		

    ## Create lists to separate fork and tag indices
    C_fork = []; C_tag = []
    mC_fork = []; mC_tag = []
    hmC_fork = []; hmC_tag = []

    ## Parse 'temp' lists fork/tag lists
    for i in xrange(10,18):
        for match in C_temp:
            if ':'+str(i) in match:
                C_fork.append(match)
        for match in mC_temp:
            if ':'+str(i) in match:
                mC_fork.append(match)
        for match in hmC_temp:
            if ':'+str(i) in match:
                hmC_fork.append(match)
                        
    for i in xrange(24,34):
        for match in C_temp:
            if str(i) in match:
                C_tag.append(match)
        for match in mC_temp:
            if str(i) in match:
                mC_tag.append(match)
        for match in hmC_temp:
            if str(i) in match:
                hmC_tag.append(match)
                

    C = np.array( map( indices.__getitem__, C_fork ) )
    mC = np.array( map( indices.__getitem__, mC_fork ) )
    hmC = np.array( map( indices.__getitem__, hmC_fork ) )

    CT = np.array( map( indices.__getitem__, C_tag ) )
    mCT = np.array( map( indices.__getitem__, mC_tag ) )
    hmCT = np.array( map( indices.__getitem__, hmC_tag ) )

    pC = np.exp( ems[ :, C ] ).sum( axis=1 )
    pmC = np.exp( ems[ :, mC ] ).sum( axis=1 )
    phmC = np.exp( ems[ :, hmC ] ).sum( axis=1 )

    pCT = np.exp( ems[ :, CT ] ).sum( axis=1 )
    pmCT = np.exp( ems[ :, mCT ] ).sum( axis=1 )
    phmCT = np.exp( ems[ :, hmCT ] ).sum( axis=1 )

    plt.subplot( 313 )
    plt.plot( pC, alpha=0.66, label='C', c='b' )
    plt.plot( pmC, alpha=0.66, label='mC', c='r' )
    plt.plot( phmC, alpha=0.66, label='hmC', c='c' )
    plt.plot( pCT, alpha=0.66, label='C-Tag', c='g' )
    plt.plot( pmCT, alpha=0.66, label='mC-Tag', c='m' )
    plt.plot( phmCT, alpha=0.66, label='hmC-Tag', c='k' )
    plt.ylabel( 'Probability' )
    plt.xlabel( 'Sequence Position' )
    plt.legend()


    plt.tight_layout()
    plt.show()	
 
 
## Partitioning and Chunkie Functions
def partition_event( indices, event, ems, means ):                      ## Fixed for substeps        
    '''
    Partitions event based on the emission matrix from the Forward-Backward Algorithm
    '''
    
    ## Find fork regions
    forks = [x for x in indices.keys() if '(C)' in x or '(mC)' in x or '(hmC)' in x ]
    forks = [x for x in forks if 'b' not in x and 'D' not in x and 'I' not in x ]
    
    ## Split into context and label fork
    C_fork = [ x for x in forks if int(x.split(':')[1]) in xrange(10, 18) ]
    L_fork = [ x for x in forks if int(x.split(':')[1]) in xrange(24, 34) ]
    
    ## Get the index values for each name in the fork / label
    C_all = np.array( map( indices.__getitem__, C_fork ) )
    L_all = np.array( map( indices.__getitem__, L_fork ) )
    
    ## Use those index values to compute probabilities by observation
    pC_all = np.exp( ems[ :, C_all ] ).sum( axis=1 )
    pL_all = np.exp( ems[:, L_all ] ).sum( axis=1 )
    
    ## Partition Events given emission matrix
    contexts, labels = [], []
    temp_c, temp_l = [], []
    
    for i in xrange(len(pC_all)): 
        ## Contexts
        if pC_all[i] > 0.5:         
            temp_c.append( i )
        if pC_all[i] <= 0.5:
            if temp_c:
                contexts.append( temp_c )
                temp_c = []
        ## Labels
        if pL_all[i] > 0.5:
            temp_l.append( i )
        if pL_all[i] <= 0.5:
            if temp_l:
                labels.append( temp_l )
                temp_l = []
    
    return contexts, labels
    
def chunk_score( indices, contexts, labels, ems ):                      ## Fixed for substeps
    ''' This function will score each context / label chunk 
        Context steps = [ 10, 12, 13, 15, 17 ] (10, 17)
        Label steps = [ 24, 26, 28, 30, 32 ] (24, 33)
    '''

    ## Find fork regions
    forks = [x for x in indices.keys() if '(C)' in x or '(mC)' in x or '(hmC)' in x ]
    forks = [x for x in forks if 'b' not in x and 'D' not in x and 'I' not in x ]
    
    ## Split into context and label fork
    C_fork = [ x for x in forks if int(x.split(':')[1]) in xrange(10, 18) ]
    L_fork = [ x for x in forks if int(x.split(':')[1]) in xrange(24, 34) ]
    
    ## Create dictionary for each state N in the fork
    '''
    c_dict, l_dict = OrderedDict(), OrderedDict()
    for i in [ 10, 12, 13, 15, 17 ]:
        c_dict[i] = np.array( map( indices.__getitem__, [x for x in C_fork if ':'+str(i) in x] ) )
    for i in [ 24, 26, 28, 30, 32 ]:
        l_dict[i] = np.array( map ( indices.__getitem__, [x for x in L_fork if ':'+str(i) in x] ) )
    '''
    
    ## Test Method 
    c_dict, l_dict = OrderedDict(), OrderedDict()
    
    c_dict[1] = np.array( map( indices.__getitem__, [x for x in C_fork if ':'+str(10) in x or ':'+str(11) in x] ) )
    c_dict[2] = np.array( map( indices.__getitem__, [x for x in C_fork if ':'+str(12) in x ] ) ) 
    c_dict[3] = np.array( map( indices.__getitem__, [x for x in C_fork if ':'+str(13) in x or ':'+str(14) in x] ) )
    c_dict[4] = np.array( map( indices.__getitem__, [x for x in C_fork if ':'+str(15) in x or ':'+str(16) in x] ) )
    c_dict[5] = np.array( map( indices.__getitem__, [x for x in C_fork if ':'+str(17) in x ] ) ) 
    
    l_dict[1] = np.array( map( indices.__getitem__, [x for x in L_fork if ':'+str(24) in x or ':'+str(25) in x] ) )
    l_dict[2] = np.array( map( indices.__getitem__, [x for x in L_fork if ':'+str(26) in x or ':'+str(27) in x] ) )
    l_dict[3] = np.array( map( indices.__getitem__, [x for x in L_fork if ':'+str(28) in x or ':'+str(29) in x] ) )
    l_dict[4] = np.array( map( indices.__getitem__, [x for x in L_fork if ':'+str(30) in x or ':'+str(31) in x] ) )
    l_dict[5] = np.array( map( indices.__getitem__, [x for x in L_fork if ':'+str(32) in x or ':'+str(33) in x] ) )
    
    ## Obtain Prior for each Context ##
    p_dict = OrderedDict()
    pscore = []
    context_final = []
    weights = [ 1.0/9, 2.0/9, 1.0/3, 2.0/9, 1.0/9 ]
    for c in contexts:
        temp_ems = ems[ c, : ]                   # Slice matrix based on observations
        for i in xrange(1, 6):
            p_dict[i] = np.max( np.exp( temp_ems[:, c_dict[i] ]).sum( axis=1 ) )
        
        ## Combine P_scores into a single score
        pscore = [ p_dict[x] for x in p_dict ] 
        pscore = [ a*b for a,b in izip(pscore, weights) ]
        pscore = sum(pscore)
        
        #if pscore > 0.9:
        context_final.append( (round(pscore,4), c) )

    ## Obtain Prior for each Label ##
    p_dict = OrderedDict()
    pscore = []
    label_final = []
    weights = [1.0/11, 2.0/11, 3.0/11, 3.0/11, 2.0/11]
    for l in labels:
        temp_ems = ems[ l, : ]
        for i in xrange(1, 6):
            p_dict[i] = np.max( np.exp( temp_ems[:, l_dict[i] ]).sum( axis=1 ) )
        
        ## Combine P_scores into a single score
        pscore = [p_dict[x] for x in p_dict]
        pscore = [a*b for a,b in izip(pscore, weights) ]
        pscore = sum(pscore)
        
        #if pscore > 0.9:
        label_final.append( (round(pscore,4), l) )
        
    return context_final, label_final
    
def chunk_vector( indices, contexts, labels, ems ):                     ## Check to make sure same method works
    
    ## Find indices for each context
    C = [ x for x in indices.keys() if '(C)' in x and 'b' not in x and 'I' not in x and 'D' not in x]
    mC = [ x for x in indices.keys() if '(mC)' in x and 'b' not in x and 'I' not in x and 'D' not in x]
    hmC = [ x for x in indices.keys() if '(hmC)' in x and 'b' not in x and 'I' not in x and 'D' not in x]
    
    ## Separate into tags and forks for each context
    C_fork = [ x for x in C if int(x.split(':')[1]) in xrange(10,18) ]
    C_tag = [ x for x in C if int(x.split(':')[1]) in xrange(24, 34) ]
    
    mC_fork = [ x for x in mC if int(x.split(':')[1]) in xrange(10,18) ]
    mC_tag = [ x for x in mC if int(x.split(':')[1]) in xrange(24, 34) ]

    hmC_fork = [ x for x in hmC if int(x.split(':')[1]) in xrange(10,18) ]
    hmC_tag = [ x for x in hmC if int(x.split(':')[1]) in xrange(24, 34) ]
    
    ## Get indices for each context / label fork
    C_fork = np.array( map( indices.__getitem__, C_fork ) )
    mC_fork = np.array( map( indices.__getitem__, mC_fork ) )
    hmC_fork = np.array( map( indices.__getitem__, hmC_fork ) )
    
    C_tag = np.array( map( indices.__getitem__, C_tag ) )
    mC_tag = np.array( map( indices.__getitem__, mC_tag ) )
    hmC_tag = np.array( map( indices.__getitem__, hmC_tag ) )
    
    ## Get a vector for each chunk
    context_final, label_final = [], []
    vector = []
    for c in contexts:
        temp_ems = ems[ c[1], : ]                       # Slice matrix based on observations
                                                        # Slice matrix by fork & sum
        vector.append( np.mean( np.exp( temp_ems[:, C_fork ]).sum( axis=1 ) ) ) 
        vector.append( np.mean( np.exp( temp_ems[:, mC_fork ]).sum( axis=1 ) ) )
        vector.append( np.mean( np.exp( temp_ems[:, hmC_fork ]).sum( axis=1 ) ) )
        
        vector = [ x*1.0/sum(vector) for x in vector ]    # Normalize list to sum to 1
        
        context_final.append( ( c[0], vector, c[1] ) )  
        vector = []
    
    for l in labels:
        temp_ems = ems[ l[1], : ]
        vector.append( np.mean( np.exp( temp_ems[:, C_tag ]).sum( axis=1 ) ) )
        vector.append( np.mean( np.exp( temp_ems[:, mC_tag ]).sum( axis=1 ) ) )
        vector.append( np.mean( np.exp( temp_ems[:, hmC_tag ]).sum( axis=1 ) ) )
        
        vector = [ x*1.0/sum(vector) for x in vector ]    # Normalize list to sum to 1
        
        label_final.append( ( l[0], vector, l[1] ) )
        vector = []
    
    return context_final, label_final


## Not working / Unneeded ?
def trans_plot( trans_dict ): 
    '''
    Creates a plot showing where certain events occur in relation to the states in the model
    
    NOT WORKING
    '''

    plt.subplot( 411 )
    plt.bar ( back[0], back[1], color='m', alpha=.66 )
    plt.title ('Backslips')
    plt.xlabel ( 'Position' )
    plt.ylabel ( 'Expected Number' )

    plt.subplot( 412 )
    plt.bar ( dis[0], dis[1], color='r', alpha=.66)
    plt.title ( 'Dissociations' )
    plt.xlabel ( 'Position' )
    plt.ylabel ( 'Expected Number' )
    plt.xlim( [0,30] )

    plt.subplot( 413 )
    plt.bar ( ins[0], ins[1], color='c', alpha=.66)
    plt.title ( 'Inserts' )
    plt.xlabel ( 'Position' )
    plt.ylabel ( 'Expected Number' )

    plt.subplot( 414 )
    plt.bar ( delete[0], delete[1], color='g', alpha=.66)
    plt.title ( 'Deletes' )
    plt.xlabel ( 'Position' )
    plt.ylabel ( 'Expected Number' )

    plt.suptitle('Number of Rereads: {}'.format( round(rereads,1) ), fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
 