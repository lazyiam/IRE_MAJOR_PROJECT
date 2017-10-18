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

wrdslst = defaultdict(dict)
# wrdslst = [defaultdict(int) for i in range(3)]
# temp =

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
            if (prevq!=lp[1] or prevtime != lp[2]) and len(lp)==5:
                wrds[lp[1]]+=1
                prevq=lp[1]
                prvtime=lp[2]
flag=0
onlyhead = defaultdict(int)
onlytail = defaultdict(int)

for i in range(1,11):
    if i!=10:
        fname = "./AOL-user-ct-collection/user-ct-test-collection-0" + str(i) + ".txt"
    else:
        fname = "./AOL-user-ct-collection/user-ct-test-collection-" + str(i) + ".txt"
    with open(fname) as FF:
        for line in FF:
            lp=line.split('\t')
            if prevq!=lp[1] or prevtime != lp[2]:
                # if wrds[lp[1]]>=500:
                #     onlyhead[lp[1]]=1
                if wrds[lp[1]]<=10 and len(lp)==5 and (lp[4].strip())!='':
                    onlytail[lp[1]]=1
                    if lp[4].strip() not in wrdslst[lp[1]]: wrdslst[lp[1]][lp[4].strip()]=0
                    wrdslst[lp[1]][lp[4].strip()]+=1
                prevq=lp[1]
                prvtime=lp[2]
ltohead = defaultdict(dict)

for i in range(1,11):
    if i!=10:
        fname = "./AOL-user-ct-collection/user-ct-test-collection-0" + str(i) + ".txt"
    else:
        fname = "./AOL-user-ct-collection/user-ct-test-collection-" + str(i) + ".txt"
    with open(fname) as FF:
        for line in FF:
            lp=line.split('\t')
            if (prevq!=lp[1] or prevtime != lp[2]) and len(lp[1])>=2:
                if len(lp)==5 and wrds[lp[1]]>=500  and (lp[4].strip())!='':
                    # onlyhead[i]=1
                    if  lp[1] not in ltohead[lp[4].strip()]:
                        ltohead[lp[4].strip()][lp[1]]=0

                    ltohead[lp[4].strip()][lp[1]]+=1
                prevq=lp[1]
                prvtime=lp[2]
# sort_lk_hd = defaultdict(dict)
wrds.clear()
for i in ltohead:
    ltohead[i] = sorted(ltohead[i].items(), key=operator.itemgetter(1))[::-1]
# ltohead.clear()
# sort_tl_lk = defaultdict(dict)
for i in wrdslst:
    wrdslst[i] = sorted(wrdslst[i].items(), key=operator.itemgetter(1))[::-1]
# wrdslst[i].clear()
tailtolink = {}
linktohead = {}
pairs = open("pairs.txt",'w')
for i in onlytail:
    # print (i)
    if wrds[i]<=10:
        # tailtolink = wrdslst[i]
        mm = 0
        for hh in wrdslst[i]:
            # print (j)
            j = hh[0]
            if mm ==0:
                prev = hh[1]
                mm =1
            curr = hh[1]
            if curr/prev < 0.6:
                break
            if curr/prev >= 0.6:
                # linktohead = ltohead[j]
                mm2=0
                for ll in ltohead[j]:
                    k = ll[0]
                    if mm2 == 0:
                        prev2 = ll[1]
                        mm2=1
                    curr2 = ll[1]
                    if curr2/prev <0.6:
                        break
                    if curr2/prev2 >=0.6:
                        trm=''
                        trm+= str(i) + " || " + str(k)
                        print(trm,file=pairs)
