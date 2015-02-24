import sys, os, random, argparse
import numpy as np
import Methods
import seaborn as sns
import matplotlib.pyplot as plt

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
with open ( '../Data/HMMs/Temp_Test.txt', 'r' ) as file:
    model = Model.read( file ) 
#print '\nTraining HMM: Witholding group {}. Training size {}. Cscore: {}'.format( i+1, len(training), cscore )
#model.train( sequences )

# Acquire indices
indices = { state.name: i for i, state in enumerate( model.states ) }

# Hold events in a list! (RANK BY EVENT)
ranked_events = []

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
                C = sorted(C, key=lambda x: x[0], reverse=True)   
                ranked_events.append( (event_name, C, L, multi, ems, means) )
                break


print '# of Events: {}'.format( len(ranked_events) )

print "Ranking Events"
ranked_events = sorted(ranked_events, key=lambda x: x[1][0][0], reverse=True )

mult_accuracy = {'f':[], 'l':[], 'r':[], 'h':[], 'i':[], 'b':[]} 
sing_accuracy = {'f':[], 'l':[], 'r':[], 'h':[], 'i':[], 'b':[]}
m_count = {} # For 2ndary X-Axis
s_count = {}
for i in xrange( 9, -1, -1):
    m_count[round(i*.1,1)] = 0
    s_count[round(i*.1,1)] = 0
   
   
for event in ranked_events:
    
    # Unpack Variables
    event_name = event[0]
    contexts = event[1]
    labels = event[2]
    multi = event[3]
    ems = event[4]
    means = event[5]
    
    barcode = event_name.split('-')[0]
    
    def run_methods(bins, count, cscore):
    
        count[cscore] += 1
        
        ## Single Read Methods
        fchunk, fcall = Methods.first_chunk( contexts, labels, cscore )
        lchunk, lcall = Methods.last_chunk( contexts, labels, cscore )
        rchunk, rcall = Methods.random_chunk( contexts, labels, cscore )

        ## Multi-Read Methods
        bchunk, bcall = Methods.best_chunk( contexts, labels )
        ichunk, icall = Methods.ind_consensus( contexts, labels, cscore )
        hchunk, hcall = Methods.hmm_consensus( indices, ems, len(means), chunk_vector )
     
        
        # First Chunk
        if barcode == fcall:
            bins['f'].append( 1 )
        else:
            bins['f'].append( 0 )

        # Last Chunk
        if barcode == lcall:
            bins['l'].append( 1 )
        else:
            bins['l'].append( 0 )

        # Random Chunk
        if barcode == rcall:
            bins['r'].append( 1 )
        else:
            bins['r'].append( 0 )

        # Best Chunk
        if barcode == bcall:
            bins['b'].append( 1 )
        else:
            bins['b'].append( 0 )

        #HMM Consensus
        if barcode == hcall:
            bins['h'].append( 1 )
        else:
            bins['h'].append( 0 )
            
        # Ind Consensus
        if barcode == icall:
            bins['i'].append( 1 )
        else:
            bins['i'].append( 0 )
                  
    cscore = float(str(contexts[0][0])[:3])
    if multi:
        run_methods(mult_accuracy, m_count, cscore)
    else:
        run_methods(sing_accuracy, s_count, cscore)


def rolling_average( acc, window=15 ): # Accepts Dict of 0/1s and produces rolling average
    '''
    Accepts dict of 0 & 1s (correct/incorrect) from ranked events 
    and produces a list of the rolling average of a given window-size.
    '''
    
    # Roling_average dict 
    ra = {'f':[], 'l':[], 'r':[], 'h':[], 'i':[], 'b':[]}
    
    for meth in acc:
        start = -window/2
        end = window/2
        for i in xrange(len(acc[meth])):
            if start < 0:
                ra[meth].append( np.mean( acc[meth][0:end] ) )
                start, end = start+1, end+1
            else:
                ra[meth].append( np.mean( acc[meth][start:end] ) )
                start, end = start+1, end+1
    
    
    return ra

def plot_rolling_average( sr, mr, s_count, m_count ):
    '''
    Plots single_read and mult_read events by rank.
    '''
    
    # Unpack Method Calls
    f_sr, f_mr = sr['f'], mr['f']
    l_sr, l_mr = sr['l'], mr['l']
    r_sr, r_mr = sr['r'], mr['r']
    h_sr, h_mr = sr['h'], mr['h']
    i_sr, i_mr = sr['i'], mr['i']
    b_sr, b_mr = sr['b'], mr['b']
    
    # Sub_plot -- Single Reads
    plt.figure(1)
    plt.subplot(211)
    plt.plot( range(1,len(f_sr)+1), f_sr, label='first', ls='--', c='k' )
    plt.plot( range(1,len(l_sr)+1), l_sr, label='last', ls='--', c='c' )
    plt.plot( range(1,len(r_sr)+1), r_sr, label='random', ls='--', c='m' )
    plt.plot( range(1,len(h_sr)+1), h_sr, label='SPD', c='g' )
    plt.plot( range(1,len(i_sr)+1), i_sr, label='IndCon', c='r', lw=2 )
    plt.plot( range(1,len(b_sr)+1), b_sr, label='Best', c='b', lw=2  )
    
    # Annotation -- Plot 1
    plt.title( 'Accuracy by Rank: Single Reads', fontsize=20 )
    plt.xlabel( 'Rank', fontsize=20 )
    plt.ylabel( 'Accuracy', fontsize=20 )
    plt.ylim( [0.0,1.0] )
    plt.xlim( [1, len(sr['f'])])
    plt.legend(loc='lower left', fontsize=14)
    
    # Sub_plot -- Multi-reads
    plt.subplot(212)
    plt.plot( range(1,len(f_mr)+1), f_mr, label='first', ls='--', c='k' )
    plt.plot( range(1,len(l_mr)+1), l_mr, label='last', ls='--', c='c' )
    plt.plot( range(1,len(r_mr)+1), r_mr, label='random', ls='--', c='m' )
    plt.plot( range(1,len(h_mr)+1), h_mr, label='SPD', c='g' )
    plt.plot( range(1,len(i_mr)+1), i_mr, label='IndCon', c='r', lw=2 )
    plt.plot( range(1,len(b_mr)+1), b_mr, label='Best', c='b', lw=2  )
    
    # Annotation -- Plot 2
    plt.title( 'Accuracy by Rank: Multi-Reads', fontsize=20 )
    plt.xlabel( 'Rank', fontsize=20 )
    plt.ylabel( 'Accuracy', fontsize=20 )
    plt.ylim( [0.0,1.0] )
    plt.xlim( [1, len(mr['f'])])
    plt.legend(loc='lower left',  fontsize=14)

    plt.tight_layout()
    plt.show()
    

print s_count, m_count    
plot_rolling_average( rolling_average(sing_accuracy), rolling_average(mult_accuracy), s_count, m_count )


