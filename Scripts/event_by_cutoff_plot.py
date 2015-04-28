#!/usr/env/python2.7
# John Vivian
#1-25-15

'''
Plot Chunk Cutoff by # of Events 
'''
from matplotlib import use
use('SVG')
import sys, ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Retrieve Events
with open('../Data/Ranked_Events/Events.txt', 'r') as f:
    Events = f.readlines()
    
# Bins -- from .999 to 0
bins = { 'leq1': [], 'leq2': [], 'leq3': [], 'leq4': [] }

## For a range of chunk cutoff, what are the number of different reads.
for i in xrange(999,-1,-1):
    i *= .001
    sys.stdout.write('Chunk Cutoff: {}\r'.format( i ))
    sys.stdout.flush()
    counter = { 'leq1': 0, 'leq2': 0, 'leq3': 0, 'leq4': 0 }
    for event in Events:
        C = event.split('@')[1]
        C = ast.literal_eval(C)
        
        C = [x for x in C if x[0] >= i]
        
        
        if len(C) == 1:
            counter['leq1'] += 1
        if 1 <= len(C) <= 2:
            counter['leq2'] += 1
        if 1 <= len(C) <= 3:
            counter['leq3'] += 1
        if len(C) >= 1:
            counter['leq4'] += 1
                
                
    for group in bins:
        bins[group].append( counter[group] )
        
def rolling_average(X):
    'computes rolling average on a list'
    new_X = []
    window = 15
    start = -window/2
    end = window/2
    for i in xrange(len(X)):
        if start < 0:
            new_X.append( np.mean( [x for x in X[0:end] if x!=0 ] ) )
            start, end = start+1, end+1
        else:
            new_X.append( np.mean( [x for x in X[start:end] if x!=0] ) )
            start, end = start+1, end+1  
    
    return new_X
        
################
#   Plotting   #
################

X = [i*.001 for i in range(999,-1,-1) ]

for i in xrange(4,0,-1):
    plt.stackplot(X, np.array(rolling_average(bins['leq'+str(i)])))  
    
ax = plt.gca()
plt.xticks(np.arange(min(X), max(X), 0.1) )
ax.invert_xaxis() 

blue = mpatches.Patch(color='blue', label='1-4+ Chunks' )
green = mpatches.Patch(color='green', label='1-3 Chunks')
red = mpatches.Patch(color='red', label='1-2 Chunks')
purple = mpatches.Patch(color='purple', label='1 Chunk')
#plt.legend(handles = [blue, green, red, purple], loc=3, fontsize=12)
plt.legend(handles = [purple, red, green, blue], loc=2, #bbox_to_anchor=(0.25, ),
          ncol=2, fancybox=True, shadow=True)

plt.xlabel("Read Cutoff", fontsize=14)
plt.ylabel("Number of Events", fontsize=14)
plt.title("Number of Events by Cutoff", fontsize=18)

#plt.show()

plt.savefig('/Users/Jvivian/Desktop/events.svg', dpi=300)


    