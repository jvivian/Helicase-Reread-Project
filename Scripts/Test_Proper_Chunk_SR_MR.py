import sys, os, random, argparse
import numpy as np
import Methods
import seaborn as sns

parser = argparse.ArgumentParser(description='Can run either simple or substep model')
parser.add_argument('-s','--substep', action='store_true', help='Imports substep model instead of simple')
args = vars(parser.parse_args())

sys.path.append( '../Models' )
if args['substep']:
    print '\n-=SUBSTEP=-'
    from Substep_Model import *
else:
    print '\n-=SIMPLE=-'
    from Simple_Model import *

## 1. Randomize 230 events into 5 groups of 46 events
# Find JSON Events 
source = '../Data/JSON/FINAL_Train/'
for root, dirnames, filenames in os.walk(source):
    events = filenames
    
# Randomize List
random.shuffle( events )

# Break into 5 equal groups
event_groups = [ events[i::5] for i in xrange(5) ]
    
## 2. For every group: withold and train on other 4. 
# Create array
data = np.zeros( (1, 12) ) 

## 3. Read in Untrained HMM then train
with open ( '../Data/HMMs/Temp_Test_2.txt', 'r' ) as file:
    model = Model.read( file ) 
#print '\nTraining HMM: Witholding group {}. Training size {}. Cscore: {}'.format( i+1, len(training), cscore )
#model.train( sequences )

# Acquire indices
indices = { state.name: i for i, state in enumerate( model.states ) }

# Rank Events by CHUNK Score
ranked_events = {}
for i in xrange(10):
    ranked_events[i] = []

print 'Ranking Events by CHUNK Score'
counter = 0
for event_name in events:  
    counter+=1
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
    
    if contexts and labels:
        max_c = max( [ x[0] for x in contexts ] ) 
        max_l = max( [ x[0] for x in labels ] )
        
        for i in xrange(9,-1,-1):
            if max_c >= i*.10 and max_l >= i*.10:
                C = [ x for x in contexts if x[0] >= i*.10 ]
                L = [ x for x in labels if x[0] >= i*.10 ]

                sys.stdout.write( 'C:{}\t\tAssigned:{}\tPercentage:{}%\r'.format(round(max_c,2), i, round((counter*1.0/len(events))*100,2)))
                sys.stdout.flush()
                
                if len(C) > 1:
                    multi=True
                else:
                    multi=False
                    
                ranked_events[i].append( (event_name, C, L, multi, ems, means) )


print '# of Events: {}'.format( len(ranked_events) )
data = { 'data_sr': np.zeros( (10, 12) ), 'data_mr': np.zeros( (10, 12) ) }

for i in ranked_events:
    print i, len(ranked_events[i])


## Iterate through the range of cutoff values: 
cscores = [ 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0 ]
data_counter = 0
for cscore in cscores:
   
    confusion = {'cm_sr': np.zeros( (3,3) ), 'cm_mr': np.zeros( ( 3,3) ) } # C, mC, hmC
    cm_choice = ['C', 'mC', 'hmC' ] # Acts as index for confusion matrix update
    # Counters
    counters = []
    counter_mr = 0
    counter_sr = 0
    cm_sr_counter = [ 0, 0, 0 ] # C, mC, hmC
    cm_mr_counter = [ 0, 0, 0 ] # C, mC, hmC
    
    # Bins to hold counts
    bins_sr = { 'f': 0, 'l': 0, 'r':0, 'b': 0, 'i': 0, 'h': 0 }
    bins_mr = { 'f': 0, 'l': 0, 'r':0, 'b': 0, 'i': 0, 'h': 0 } # Counter for hard calls
    soft_calls = { 'f': [], 'l': [], 'r':[], 'b': [], 'i': [], 'h': [] }    # Will hold soft calls
    
    ## For a given cscore, group and iterate through.
    event_sum = 0

    for event in ranked_events[int(cscore*10)]:
    
        # Unpack Variables
        event_name = event[0]
        contexts = event[1]
        labels = event[2]
        multi = event[3]
        ems = event[4]
        means = event[5]
       
        barcode = event_name.split('-')[0]
       
        def run_methods(bins, cm, cm_counter ):
        
            ## Single Read Methods
            fchunk, fcall = Methods.first_chunk( contexts, labels, cscore )
            lchunk, lcall = Methods.last_chunk( contexts, labels, cscore )
            rchunk, rcall = Methods.random_chunk( contexts, labels, cscore )

            ## Multi-Read Methods
            bchunk, bcall = Methods.best_chunk( contexts, labels )
            ichunk, icall = Methods.ind_consensus( contexts, labels, cscore )
            hchunk, hcall = Methods.hmm_consensus( indices, ems, len(means), chunk_vector )
         
            
            # First Chunk
            #soft_calls['f'].append( fchunk )
            if barcode == fcall:
                bins['f'] += 1

            # Last Chunk
            #soft_calls['l'].append( lchunk )
            if barcode == lcall:
                bins['l'] += 1

            # Random Chunk
            #soft_calls['r'].append( rchunk )
            if barcode == rcall:
                bins['r'] += 1

            # Best Chunk
            #soft_calls['b'].append( bchunk )
            if barcode == bcall:
                bins['b'] += 1

            #HMM Consensus
            #soft_calls['h'].append( hchunk )
            if barcode == hcall:
                bins['h'] += 1
                
            # Ind Consensus
            #soft_calls['i'].append( ichunk )
            if barcode == icall:
                bins['i'] += 1
            
            ## Confusion Matrix

            cm_counter[ cm_choice.index( barcode) ] += 1 # Update appropriate counter
            cm[ cm_choice.index( barcode ), cm_choice.index( icall ) ] += 1 # Updates correct row given bar code
            
                
                
        if multi:
            run_methods(bins_mr, confusion['cm_mr'], cm_mr_counter )
            # update counter
            counter_mr += 1
        else:
            run_methods(bins_sr, confusion['cm_sr'], cm_sr_counter)
            counter_sr += 1
    
    print '# of MR events:{}\t# of SR events:{}\tcsore:{}'.format( counter_mr, counter_sr, cscore )
    j = data_counter
    for i in data:
        if i == 'data_mr':
            data[i][j][0] = bins_mr['f']*1.0 / counter_mr
            data[i][j][1] = np.mean(soft_calls['f'])
            data[i][j][2] = bins_mr['l']*1.0 / counter_mr
            data[i][j][3] = np.mean(soft_calls['l'])
            data[i][j][4] = bins_mr['r']*1.0 / counter_mr
            data[i][j][5] = np.mean(soft_calls['r'])
            data[i][j][6] = bins_mr['h']*1.0 / counter_mr
            data[i][j][7] = np.mean(soft_calls['h'])
            data[i][j][8] = bins_mr['b']*1.0 / counter_mr
            data[i][j][9] = np.mean(soft_calls['b'])
            data[i][j][10] = bins_mr['i']*1.0 / counter_mr
            data[i][j][11] = np.mean(soft_calls['i'])
            
            ## CM
            for i in xrange(3):
                confusion['cm_mr'][i] /= cm_mr_counter[i]
            print '-=Multi-Read Confusion Matrix=-'
            print 'C:{}\tmC:{}\thmC:{}'.format( cm_mr_counter[0], cm_mr_counter[1], cm_mr_counter[2] )
            print 'C\tmC\thmC\n{}'.format( confusion['cm_mr'] )
            
        else:
            data[i][j][0] = bins_sr['f']*1.0 / counter_sr
            data[i][j][1] = np.mean(soft_calls['f'])
            data[i][j][2] = bins_sr['l']*1.0 / counter_sr
            data[i][j][3] = np.mean(soft_calls['l'])
            data[i][j][4] = bins_sr['r']*1.0 / counter_sr
            data[i][j][5] = np.mean(soft_calls['r'])
            data[i][j][6] = bins_sr['h']*1.0 / counter_sr
            data[i][j][7] = np.mean(soft_calls['h'])
            data[i][j][8] = bins_sr['b']*1.0 / counter_sr
            data[i][j][9] = np.mean(soft_calls['b'])
            data[i][j][10] = bins_sr['i']*1.0 / counter_sr
            data[i][j][11] = np.mean(soft_calls['i'])
            
            ## CM
            for i in xrange(3):
               confusion['cm_sr'][i] /= cm_sr_counter[i]
            #print cm_sr_counter, '\n', confusion['cm_sr']
            print '-=Single-Read Confusion Matrix=-'
            print 'C:{}\tmC:{}\thmC:{}'.format( cm_sr_counter[0], cm_sr_counter[1], cm_sr_counter[2] )
            print 'C\tmC\thmC\n{}'.format( confusion['cm_sr'] )
            
    data_counter += 1
    
    
    
def accuracy_by_filter_score( data, title, sc=False ):
    
    # Convert Groups of arrays into lists of scores by csore.
    first, last, random = [], [], []
    hmm, best, ind = [], [], []
    sample_sizes = []
    
    for j in xrange(0,10):
            
            #trial = np.loadtxt( '../Data/Results/' + trial_name, delimiter = ',' )
            #accuracies = means[::2]
            #acc_std = stds[::2]
            #softcalls = means[1::2]
            #sc_std = stds[1::2]
            
            trial = data[j]
        
            accuracies = trial[::2]
            softcalls = trial[1::2]
            
            if sc:
                f,l,r,h,b,i = softcalls
            else:
                f,l,r,h,b,i = accuracies
            first.append( f ); last.append( l ); random.append( r )
            hmm.append( h ) ; best.append( b ); ind.append( i )
         
            #sample_sizes.append( trial_name.split('_')[3] )

    x = [ 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0 ]
    plt.plot( x, first, label='first', ls='--', c='k')
    plt.plot( x, last, label='last', ls='--', c='c') 
    plt.plot( x, random, label='random', ls='--', c='m' )
    
    plt.plot( x, hmm, label='hmm', c='y' )
    plt.plot( x, best, label='best', lw=2, c='r' )
    plt.plot( x, ind, label='ind', lw=2, c='b')
    
   # plt.title( trial_name.split('_')[0] + ' - ' + trial_name.split('_')[1] \
    #            + ' - ' + trial_name.split('_')[2], fontsize=14 )
    
    plt.title( title, fontsize=20 )
    plt.xlabel( 'Chunk Score Cutoff', fontsize=20 )
    plt.ylabel( 'Accuracy', fontsize=20 )
    plt.ylim( [0.5,1.0] )
    plt.legend(loc='lower left', fontsize=14)
    plt.gca().invert_xaxis()
    plt.show()
    
accuracy_by_filter_score( data['data_sr'], 'Single Reads: Alpha=1.5' )
accuracy_by_filter_score( data['data_mr'], 'Multi-Reads: Alpha=1.5' )
