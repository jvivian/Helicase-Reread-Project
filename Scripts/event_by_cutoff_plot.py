#!/usr/env/python2.7
# John Vivian
#1-25-15

'''
Plot for # of events 
'''
import sys, ast
import numpy as np
import Methods
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Retrieve Events
with open('..\Data\Ranked_Events\Events.txt', 'r') as f:
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
        
        C = [x for x in C if x[0] > i]
        
        
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
        

        
################
#   Plotting   #
################

X = [i*.001 for i in range(999,-1,-1) ]

for i in xrange(4,0,-1):
    plt.stackplot(X, np.array(bins['leq'+str(i)]))  
    
ax = plt.gca()
ax.invert_xaxis() 

blue = mpatches.Patch(color='blue', label='1-4+ Chunks' )
green = mpatches.Patch(color='green', label='1-3 Chunks')
red = mpatches.Patch(color='red', label='1-2 Chunks')
purple = mpatches.Patch(color='purple', label='1 Chunk')
plt.legend(handles = [blue, green, red, purple], loc=3, fontsize='large')   

plt.xlabel("Chunk Cutoff", fontsize=14)
plt.ylabel("Number of Events", fontsize=14)
plt.title("Number of Events by Chunk Cutoff", fontsize=20)

plt.show()




    