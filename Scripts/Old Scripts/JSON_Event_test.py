#!usr/bin/env python2.7
# John Vivian

import sys, os

sys.path.append( '../Models' )

from Simple_Model import *

## Find .JSONS
source = '../Data/JSON'
files = []
for root, dirnames, filenames in os.walk(source):
    files = filenames


events = [ Event.from_json( '../Data/JSON/' + x ) for x in files ]

print len(events)