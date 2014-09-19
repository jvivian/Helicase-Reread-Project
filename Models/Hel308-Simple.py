#!usr/bin/env 
# John Vivian

'''
Hel308-Simple.py is a collection of functions that perform the following operations:
    
    1.  Builds a profile from an external data file
    2.  Builds a Hidden Markov Model (HMM) from the profile data
    3.  Parses an .ABF file (nanopore data) in order to iterate through discrete events
    4.  Events are passed through the HMM to return a comprehensive Viterbi path.
    5.  Additional information culled from the Forward-Backward algorithm is also displayed
    6.  Plots the segmented event, the event as colored by HMM-board, and the emission probabilities 
        for the fork and label region
    
Special thanks to Jacob Schreiber for PyPore and his contributions to YAHMM
'''

from yahmm import *
import pandas as pd
from PyPore.DataTypes import *
import sys

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

def build_profile( ):
    '''
    Reads in an excel to obtain profile for the HMM.  
    Profile is stored as a list of distributions with forks represented by dictionaries.
    '''
    profile, dists = [], {}
    data = pd.read_excel( '../profile/CCGG.xlsx', 'Sheet3' )

    total_means, total_stds = data.mean(axis=0), data.std(axis=0) # Total Profile Man
    total = [ NormalDistribution( m, 1.5 ) for m, s in zip( total_means, total_stds) ]

    for name, frame in data.groupby('label'):
        means, stds = frame.mean(axis=0), frame.std(axis=0)
        dists[name] = [ NormalDistribution( m, s ) for m, s in zip( means, stds )]

    # Piecing the profile together (0-3) = states 1-4
    for i in xrange(6):
        profile.append(total[i])

    # Cytosine fork (6-10) = states 7-11
    for i in xrange(6, 11):
        profile.append( {'C': dists['C'][i], 'mC': dists['mC'][i], 'hmC': dists['hmC'][i]  })

    # Continuation of profile  (11-15) = states 12-16
    for i in xrange(11, 16):
        profile.append(total[i])

    # Label Fork (16-20) = states 17-21
    for i in xrange(16,21):
        profile.append( {'C': dists['C'][i], 'mC': dists['mC'][i], 'hmC': dists['hmC'][i]  })

    # Continuation of profile  (21-27) = states 22-28
    for i in xrange(21, 32):
        profile.append(total[i])
        
    fourmers = [ col.replace(' ', '_').replace('.1', '').replace('.2', '') for col in data ][1:] 
    fourmers = [a.encode('ascii', 'ignore') for a in fourmers]

    return profile, fourmers
    
def parse_abf(abf, start=0, end=750):
    '''
    parses an abf file and yields an event
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
   
def viterbi(model, event, fourmers):
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
    for i in xrange(7,12):  # (6-10) = (7-11) because of I0 insert state.
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

    #############
    #	Output	#
    #############

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

    return { 'd' : d_trans, 'i' : i_trans, 'b1': b1_trans, 'b2': b2_trans, 'b3' : b3_trans, \
            'b4' : b4_trans, 'b5' : b5_trans, 'dis' : dis_trans, 'r' : r_trans }
    
def segment_ems_plot( model, event, ems):
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
    for i in xrange(7,12):
        for match in C_temp:
            if ':'+str(i) in match:
                C_fork.append(match)
        for match in mC_temp:
            if ':'+str(i) in match:
                mC_fork.append(match)
        for match in hmC_temp:
            if ':'+str(i) in match:
                hmC_fork.append(match)
                        
    for i in xrange(17,22):
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
 
def trans_plot( trans_dict ): ## Not working 
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
 
print '\n-=Building Profile=-'
distributions, fourmers = build_profile()
'''
print '-=Building HMM=-'
model = Hel308_simple_model( distributions, 'Test-31', fourmers )

print '-=Parsing ABF=-'
for event in parse_abf('..\..\abfs\\Mixed\\14710002-s01.abf', 735):
    
    print '-=Determining Viterbi Path=-'
    viterbi(model, event, fourmers)
    
    ## Perform the forward_backward algorithm to return transmission and emission matrices
    means = [seg.mean for seg in event.segments]
    trans, ems = model.forward_backward( means )
    
    ## Summary Information 
    trans_dict = analyze_event( model, event, trans )

    ## Plots
    segment_ems_plot( model, event, ems)   
    
    This is how to SAVE the HMM after you've created it
    with open( 'test.txt', 'w' ) as file:
        model.write( file )
    '''
'''
Change-Log (Semantic Versioning:  Major-Minor-Patch)

Version 0.1.0       -       9-18-14
    1. Initial Commit
'''
