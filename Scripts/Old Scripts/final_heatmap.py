__author__ = 'Jvivian'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame(
    [[ 0.76,  0.04,  0.20],
 [ 0.32,      0.48,        0.20],
 [ 0.21,  0.05,  0.74]] )


## Plotting
# setup
fig, ax = plt.subplots()
plt.pcolor(df, cmap='Oranges')

# Put the major ticks in the middle of each cell
ax.set_xticks(np.arange(df.shape[1]) +.5, minor=False)
ax.set_yticks(np.arange(df.shape[0]) +.5, minor=False)
ax.invert_yaxis()
plt.xticks(rotation=50)

# Labels
row_labels = ['C', 'mC', 'hmC']
col_labels = row_labels

ax.set_xticklabels(col_labels, minor=False)
ax.set_yticklabels(row_labels, minor=False)
#plt.gca().set_xlim(0, 18)      # Context
plt.title( 'Confusion Matrix' )

for y in range(df.shape[0]):
    for x in range(df.shape[1]):
        #print x, y
        plt.text(x + 0.5, y + 0.5, '{}'.format(df[x][y]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 )

cb = plt.colorbar()

#plt.show()
plt.savefig('/Users/Jvivian/Desktop/CM.png', dpi=300)



