import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

x = [0.005, 0.00255, 0.00128, 0.0006365, 0.0003159, 0.005, 0.00255, 0.00128, 0.0006365, 0.0003159 ]
plt.plot ( x, linestyle='--', marker='o', color='b')
labels = [str(x) for x in xrange(1,11)]
plt.ylim( [0, 0.006] )
plt.ylabel('Probability')
plt.xlabel('Length of Backslip')
plt.xticks(np.arange(len(labels)), labels)
plt.title('Probability of Backslips at Different Lengths')
plt.show()
plt.savefig('/Users/Jvivian/Desktop/backslip.svg')