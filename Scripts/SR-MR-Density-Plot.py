#!/usr/bin/env python2.7
# John Vivian
# 1-25-15

'''
At every point between [0,1] (0.001 interval), the accuracies for multi and single reads
are averaged. The plots are then smoothed with a rolling average.
'''


import sys, ast
import numpy as np
import Methods
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def return_hardcall(C,L,cutoff,barcode,Ind=False):

    bchunk, bcall = Methods.best_chunk( C, L )
    ichunk, icall = Methods.ind_consensus( C, L, cutoff )
    
    if Ind:
        if barcode == icall:
            return 1
        else:
            return 0
    else:
        if barcode == bcall:
            return 1
        else:
            return 0


# Retrieve Events
with open('../Data/Ranked_Events/Events.txt', 'r') as f:
    Events = f.readlines()

SR_Master = [] # Will hold 1000 averaged points for single reads
MR_Master_Ind = []
MR_Master_Best = []

for cutoff in xrange(999,-1,-1):
    cutoff *= .001

    sys.stdout.write("Counter: {}\r".format( cutoff ))
    sys.stdout.flush()

    sr_average = [] # Vectors that will hold values at this cutoff
    mr_average_Ind = []
    mr_average_Best = []

    for event in Events:
        C = event.split('@')[1] # Unpack Contexts 
        C = ast.literal_eval(C) # Cast string as a list

        L = event.split("@")[2] 
        L = ast.literal_eval(L)

        barcode = event.split('@')[0].split('-')[0] # Correct Call for Event
        

        C = [x for x in C if x[0] >= cutoff] # Filter chunks by cutoff

        if len(C) == 1:
            sr_average.append( return_hardcall(C,L,cutoff,barcode) )

        elif len(C) > 1:
            mr_average_Ind.append( return_hardcall(C,L,cutoff,barcode,True) )
            mr_average_Best.append( return_hardcall(C,L,cutoff,barcode) )

    if sr_average:
        SR_Master.append( np.mean( sr_average) )
    else:
        SR_Master.append(np.nan)


    if mr_average_Ind:
        MR_Master_Ind.append( np.mean( mr_average_Ind) )
    else:
        MR_Master_Ind.append( np.nan )
    
    if mr_average_Best: 
       MR_Master_Best.append( np.mean( mr_average_Best) )
    else:
        MR_Master_Best.append( np.nan )


def rolling_average(X):
    '''takes in list and returns list after performing rolling average '''
    new_X = []
    std = []
    window = 15
    start = -window/2
    end = window/2
    for i in xrange(len(X)):
        if start < 0:
            new_X.append( np.mean( [x for x in X[0:end] if x!=0 ] ) )
            std.append( np.std( [x for x in X[0:end] if x!=0 ] ) )
            start, end = start+1, end+1
        else:
            new_X.append( np.mean( [x for x in X[start:end] if x!=0] ) )
            std.append( np.std( [x for x in X[start:end] if x!=0 ] ) )
            start, end = start+1, end+1  
    
    return new_X, std
    
#################
#   Plotting    #
#################

X = [i*.001 for i in range(999,-1,-1) ]

SR_Master, SR_std = rolling_average(SR_Master)
MR_Master_Best, MR_std = rolling_average(MR_Master_Best)
MR_Master_Ind, MRI_std = rolling_average(MR_Master_Ind)

ttest = stats.ttest_ind([i for i in SR_Master if i > 0], [i for i in MR_Master_Best if i > 0])
print ttest

SR_plus = []
SR_minus = []
for i in xrange(len(SR_Master)):
    SR_plus.append( SR_Master[i]+SR_std[i] )
    SR_minus.append( SR_Master[i]-SR_std[i] )

MR_plus = []
MR_minus = []
for i in xrange(len(MR_Master_Best)):
    MR_plus.append( MR_Master_Best[i]+MR_std[i] )
    MR_minus.append( MR_Master_Best[i]-MR_std[i] )

SR_mean = np.mean([i for i in SR_Master if i > 0])
MR_mean = np.mean([i for i in MR_Master_Best if i > 0])

SR_mean_plot = [SR_mean for i in xrange(1000)]
MR_mean_plot = [MR_mean for i in xrange(1000)]

plt.plot(X, SR_mean_plot, label="SR Average", lw=2, ls='--')
plt.plot(X, MR_mean_plot, label='MR Average', lw=2, ls='--')

plt.plot(X, SR_Master, label='Single Reads', lw=2 )
plt.plot(X, MR_Master_Ind, label='Independent Consensus', lw=2)
plt.plot(X, MR_Master_Best, label='Best Consensus', lw=2)
ax = plt.gca()
plt.xticks(np.arange(min(X), max(X), 0.1) )
ax.set_xlim([0.75, 0.95])
ax.invert_xaxis() 
ax.set_ylim([0.6,0.85])
#ax.fill_between(X, SR_plus, SR_minus, alpha=0.5 )
#ax.fill_between(X, MR_plus, MR_minus, alpha=0.5)

plt.title('Accuracies of Single vs. Multiple Reads', fontsize=20)
plt.xlabel('Chunk Cutoff', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(fontsize='large', loc=3)
plt.show()

