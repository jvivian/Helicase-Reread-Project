#!/usr/bin/env python2.7
# John Vivian

'''
Script for creating a heatmap that shows which of the states in the model
are most important.
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

################################

total_diff = []
dist = {}

## Read in data to a DataFrame
data = pd.read_excel( '../Profile/CCGG.xlsx', 'Sheet3')

## Split DataFrames by label and store the means of each as a Series
for name, frame in data.groupby( 'label' ):
    dist[name] = frame.mean(axis=0)

## Create comparisons for each context
C_mC = np.abs( dist['C'] - dist['mC'] ).tolist()
mC_hmC = np.abs( dist['mC'] - dist['hmC'] ).tolist()
C_hmC = np.abs( dist['C'] - dist['hmC'] ).tolist()

## Create total comparisons 
total = zip( dist['C'], dist['mC'], dist['hmC'] )
for item in total:
    total_diff.append( abs( np.max(item) - np.min(item) ) )

## Assemble Final DataFrame
#df = pd.DataFrame( [ C_mC[:16], mC_hmC[:16], C_hmC[:16], total_diff[:16] ] )   # Context
df = pd.DataFrame( [ C_mC[10:], mC_hmC[10:], C_hmC[10:], total_diff[10:] ] )    # Label

## Plotting
# setup
fig, ax = plt.subplots()
plt.pcolor(df, cmap='Reds')

# Put the major ticks in the middle of each cell
ax.set_xticks(np.arange(df.shape[1]) +.5, minor=False)
ax.set_yticks(np.arange(df.shape[0]) +.5, minor=False)
ax.invert_yaxis()
plt.xticks(rotation=50)

# Labels
row_labels = ['C_mC', 'mC_hmC', 'C_hmC', 'Max $\Delta$ ']
#column_labels = data.columns[1:17]     # Context
column_labels = data.columns[11:]       # Label
column_labels = [x.replace('.1','').replace('.2','') for x in column_labels]
ax.set_xticklabels(column_labels, minor=False)
ax.set_yticklabels(row_labels, minor=False)
#plt.gca().set_xlim(0, 16)      # Context
plt.gca().set_xlim(0, 22)       # Label
plt.title( 'State Importance' )

cb = plt.colorbar()
cb.set_label('Absolute Mean Difference', labelpad=20, fontsize='16')
plt.show()

'''
Change-Log (Semantic Versioning:  Major-Minor-Patch)

Version 0.1.0       -       9-19-14
    1. Initial Commit
'''

