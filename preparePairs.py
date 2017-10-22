import os
import sys
import numpy as np

#Getting command line input
inputFile = sys.argv[1]
fileLabel = sys.argv[2]
vectorFile = sys.argv[3]
outputFile = sys.argv[4]

#Loading vector keys
vectors = {}
vecFile = open(vectorFile)
for line in vecFile:
	lineParts = line.split()
	vectors[lineParts[0]] = ''

#Reading stop words for processing queries
stopWords = {}
f = open('enSmall.txt')
for line in f:
	stopWords[line.strip()] = ''
f.close()

#Reading the input file and preparing Pair file to be input to NN code
iterCount = 0
termList = {}
f = open(inputFile)
fw = open(outputFile, 'w')
for line in f:
	iterCount += 1
	if(iterCount % 100000 == 0):
		print iterCount
	
	lineParts = line.split(' || ')
	tq = lineParts[0].strip()
	hq = lineParts[1].strip()
	
	#Processing stopWords and collecting termlist
	newTq = ''
	for word in tq.split():
		if(word in stopWords):
			continue
		else:
			newTq = newTq + ' ' + word
			if(word not in termList):
				termList[word] = ''
	newTq = newTq.strip()
	if(len(newTq.strip()) == 0):
		continue
		
	newHq = ''
	for word in hq.split():
		if(word in stopWords):
			continue
		else:
			newHq = newHq + ' ' + word
			if(word not in termList):
				termList[word] = ''
	newHq = newHq.strip()
	if(len(newHq.strip()) == 0):
		continue

	#Modifying queries to their 10 word (tail query) and 5 word (head query) versions 
	#Also replacing missing/non-existent words with special term ~ZERO~
	tq = ''
	hq = ''

	tqLen = len(newTq.split())
	if(tqLen > 10):
		i = 0
		for word in newTq.split():
			i = i + 1
			if(i > 10):
				break
			
			if(word in vectors):
				tq = tq + ' ' + word
			else:
				tq = tq + ' ~ZERO~'
	else:
		for word in newTq.split():
			if(word in vectors):
				tq = tq + ' ' + word
			else:
				tq = tq + ' ~ZERO~'
		
		for i in range(10 - tqLen):
			tq = tq + ' ~ZERO~'
	tq = tq.strip()
	
	hqLen = len(newHq.split())
	if(hqLen > 5):
		i = 0
		for word in newHq.split():
			i = i + 1
			if(i > 5):
				break
			
			if(word in vectors):
				hq = hq + ' ' + word
			else:
				hq = hq + ' ~ZERO~'
	else:
		for word in newHq.split():
			if(word in vectors):
				hq = hq + ' ' + word
			else:
				hq = hq + ' ~ZERO~'
		
		for i in range(5 - hqLen):
			hq = hq + ' ~ZERO~'
	hq = hq.strip()
		
	fw.write(tq + '\t' + hq + '\t' + fileLabel + '\n')
fw.close()

#Print the termList to get vectors from the main file
f = open('termList.txt', 'w')
for term in termList:
	f.write(term + '\n')
f.close()
