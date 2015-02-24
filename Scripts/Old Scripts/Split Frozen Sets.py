#!usr/bin/env python2.7
# John Vivian

'''
Small script to break up JSON's into random groups
'''

import random, os, shutil

# Find JSON Events 
source = '../Data/JSON/Full'
for root, dirnames, filenames in os.walk(source):
    events = filenames

# Randomize List
random.shuffle( events )

# Break into equal groups
event_groups = [ events[i::3] for i in xrange(3) ]
for i in event_groups:
	print len(i)

## Copy from source directory to frozen directory
for jsons in event_groups[0]:
	shutil.copy2( '../Data/JSON/Full/' + str(jsons), '../Data/JSON/Frozen_1')

for jsons in event_groups[1]:
	shutil.copy2( '../Data/JSON/Full/' + str(jsons), '../Data/JSON/Frozen_2')

for jsons in event_groups[2]:
	shutil.copy2( '../Data/JSON/Full/' + str(jsons), '../Data/JSON/Frozen_3')
