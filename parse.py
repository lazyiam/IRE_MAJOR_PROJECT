from __future__ import print_function
import codecs
import csv,sys
import heapq
import time
import os
import re
import math
import operator
import threading
from collections import *
cnt =0
wrds =defaultdict(int)
fname = ''
prevq= ''
prevtime=''
for i in range(1,11):
    if i!=10:
        fname = "./AOL-user-ct-collection/user-ct-test-collection-0" + str(i) + ".txt"
    else:
        fname = "./AOL-user-ct-collection/user-ct-test-collection-" + str(i) + ".txt"
    with open(fname) as FF:
        for line in FF:
            lp=line.split('\t')
            if prevq!=lp[1] or prevtime != lp[2]:
                wrds[lp[1]]+=1
                prevq=lp[1]
                prvtime=lp[2]
tail=open("tail.txt",'w')
head=open("head.txt",'w')
for i in wrds:
    trm=""
    trm += str(i) + " " + str(wrds[i])
    if wrds[i]<50 :
        print(trm,file=tail)
    elif wrds[i]>=500:
        print(trm,file=head)
