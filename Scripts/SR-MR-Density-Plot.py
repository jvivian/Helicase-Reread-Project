
# Define each event as a function x = chunk cutoff [.999, 0]. 
#
#  Event  = [.99, .90, .44 ]
#
#
#  sr = [ (.99, 1), (.98, 1 ) ... (.91, 1) ]   # If Correct 
#  mr = [ (.90, 0), (.89, 0), ... (.44, 1) ... ] # If first addition incorrect, section correct

'''
Pseudo-code

SR_FUNCTIONS = [] # Will store our our list of functions
MR_FUNCTIONS = [] # Will store the multi list of functions

for event in EVENTS:
   
   SINGLE_READ_PARTITION = []
   MULTI_READ_PARTITION = []
   
    for i in xrange(999,-1,-1):
        i *= .001
        
        C = [x for x in CONTEXT if x[0] >= i]
        
        if len(C) == 1:
            
            CHECK if correct
            STORE tuple of x/y (i, {0/1}) in SINGLE_READ_PARTITION
            
        if len(C) > 1:
            
            CHECK if correct
            STORE TUPLE of x/y (i, {0/1}) in MULTI_READ_PARTITION
                

# Merge functions ???
            
# Dynamically plot all that shit. 

... Alternatively


For each POINT average the correct number of SR and MR....  Exception statement for 0s. 
'''


import sys, ast
import numpy as np
import Methods
import matplotlib.pyplot as plt
import seaborn as sns

def return_hardcall(C,L,cutoff,barcode):

    bchunk, bcall = Methods.best_chunk( C, L )
    ichunk, icall = Methods.ind_consensus( C, L, cutoff )

    if barcode == icall:
        return 1
    else:
        return 0


# Retrieve Events
with open('..\Data\Ranked_Events\Events.txt', 'r') as f:
    Events = f.readlines()

SR_Master = [] # Will hold 1000 averaged points for single reads
MR_Master = [] # Will hold 1000 averaged points for multi reads 

for cutoff in xrange(999,-1,-1):
    cutoff *= .001

    sys.stdout.write("Counter: {}\r".format( cutoff ))
    sys.stdout.flush()

    sr_average = [] # Vectors that will hold values at this cutoff
    mr_average = [] 

    for event in Events:
        C = event.split('@')[1] # Unpack Contexts 
        C = ast.literal_eval(C) # Cast string as a list

        L = event.split("@")[2]
        L = ast.literal_eval(L)

        barcode = event.split('@')[0].split('-')[0]
        

        C = [x for x in C if x[0] >= cutoff] # Filter chunks by cutoff

        if len(C) == 1:
            sr_average.append( return_hardcall(C,L,cutoff,barcode) )

        elif len(C) > 1:
            mr_average.append( return_hardcall(C,L,cutoff,barcode) )

    if sr_average:
        SR_Master.append( np.mean( sr_average) )
    else:
        SR_Master.append(0)

    if mr_average:
        MR_Master.append( np.mean( mr_average) )
    else:
        MR_Master.append(0)


#################
#   Plotting    #
#################

X = [i*.001 for i in range(999,-1,-1) ]

plt.plot(X, SR_Master, label='Single Reads', lw=2 )
plt.plot(X, MR_Master, label='Multiple Reads', lw=2)

ax = plt.gca()
ax.invert_xaxis() 
ax.set_ylim([0.5,1])

plt.title('Density of Single vs. Multiple Reads', fontsize=20)
plt.xlabel('Chunk Cutoff', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(fontsize='large')
plt.show()

