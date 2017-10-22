import sys
import math
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from keras.models import Model
from keras.layers import Input, merge, concatenate, LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten

vectorFile = sys.argv[1]
trainFile = sys.argv[2]
numTrain = int(sys.argv[3])

#Loading vectors
print ('Loading Vectors')
vectors = {}
vecFile = open(vectorFile)
for line in vecFile:
	lineParts = line.split()
	temp = np.asarray(lineParts[1:])
	try:
		vector = temp.astype(np.float)
		vectors[lineParts[0]] = vector
	except:
		print 'Not loading: ' + lineParts[0]
		pass

#Reading training file
"""print ('Loading Train Data')
trFile = open(trainFile)
X = []
for line in trFile:
	line = line.decode('utf-8')
	lineParts = line.split('\t')
	X.append((lineParts[0], lineParts[1], lineParts[2]))
trFile.close()"""

posTotal = 0.2 * numTrain
negTotal = 0.8 * numTrain
print ('Loading Train Data')
trFile = open(trainFile)
X = []
posCount = 0
negCount = 0
for line in trFile:
	line = line.decode('utf-8').strip()
	lineParts = line.split('\t')

	if(lineParts[2] == '1' and posCount < posTotal):
		X.append((lineParts[0], lineParts[1], lineParts[2]))
		posCount += 1
	
	if(lineParts[2] == '0' and negCount < negTotal):
		X.append((lineParts[0], lineParts[1], lineParts[2]))
		negCount += 1

	if(posCount == posTotal and negCount == negTotal):
		break
trFile.close()
print str(len(X))

#Shuffling the input so that positive and negative examples are disarranged
random.shuffle(X)

#Splitting into training and test sets (80-20 split)
splitTrainEnd = int(math.ceil(0.8 * numTrain))
splitTestEnd = numTrain
#partVal = 0.0005
#splitTrainEnd = int(math.ceil(partVal * numTrain))
#splitTestEnd = splitTrainEnd + (splitTrainEnd * 0.2)

zeroVec = np.zeros((300,))
X_train_tq = np.empty((splitTrainEnd, 10, 300))
X_train_hq = np.empty((splitTrainEnd, 5, 300))
Y_train = np.empty((splitTrainEnd, 1))
pos = 0
neg = 0
for i in range(splitTrainEnd):
	try:
		tq = X[i][0]
	except:
		print str(i)
	innerArr = np.empty((10, 300))
	tqParts = tq.split(' ')
	for j in range(len(tqParts)):
		if(tqParts[j] == '~ZERO~' or tqParts[j] not in vectors):
			innerArr[j] = zeroVec
		else:
			innerArr[j] = vectors[tqParts[j]]
	X_train_tq[i] = innerArr
	
	hq = X[i][1]
	innerArr = np.empty((5, 300))
	hqParts = hq.split() 
	for j in range(len(hqParts)):
		if(hqParts[j] == '~ZERO~' or hqParts[j] not in vectors):
			innerArr[j] = zeroVec
		else:
			innerArr[j] = vectors[hqParts[j]]
	X_train_hq[i] = innerArr
	
	Y_train[i] = float(X[i][2])
	if(Y_train[i] == 1.0):
		pos += 1
	else:
		neg += 1
print ('Training - Positive: ' + str(pos) + ' Negative: ' + str(neg))

numTestInst = splitTestEnd - splitTrainEnd
X_test_tq = np.empty((numTestInst, 10, 300))
X_test_hq = np.empty((numTestInst, 5, 300))
Y_test = np.empty((numTestInst, 1))
pos = 0
neg = 0
for i in range(numTestInst):
	k = i + splitTrainEnd
	tq = X[k][0]
	innerArr = np.empty((10, 300))
	tqParts = tq.split(' ')
	for j in range(len(tqParts)):
		if(tqParts[j] == '~ZERO~' or tqParts[j] not in vectors):
			innerArr[j] = zeroVec
		else:
			innerArr[j] = vectors[tqParts[j]]
	X_test_tq[i] = innerArr
	
	hq = X[k][1]
	innerArr = np.empty((5, 300))
	hqParts = hq.split(' ') 
	for j in range(len(hqParts)):
		if(hqParts[j] == '~ZERO~' or hqParts[j] not in vectors):
			innerArr[j] = zeroVec
		else:
			innerArr[j] = vectors[hqParts[j]]
	X_test_hq[i] = innerArr

	Y_test[i] = float(X[k][2])
	if(Y_test[i] == 1.0):
		pos += 1
	else:
		neg += 1
print ('Testing - Positive: ' + str(pos) + ' Negative: ' + str(neg))

print ('Data sizes:')
print str(X_train_hq.shape) + '\t' + str(Y_train.shape) + '\t' + str(X_test_hq.shape) + '\t' + str(Y_test.shape)

epos = 1
batchSize = 128

print ('Creating NN Graph model')
tqInp = Input(shape=(10, 300), name='tqInput')
hqInp = Input(shape=(5, 300), name='hqInput')

tqLSTM = LSTM(100, activation='relu')(tqInp)
hqLSTM = LSTM(100, activation='relu')(hqInp)

midMerge = concatenate([tqLSTM, hqLSTM])
out = Dense(1, activation='sigmoid')(midMerge)

model = Model(inputs=[tqInp, hqInp], outputs=[out])
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

model.summary()

#Training model
print ('Training')
model.fit([X_train_tq, X_train_hq], [Y_train], epochs = epos, batch_size = batchSize)

#Testing
print ('Testing')
Y_pred = model.predict({'tqInput': X_test_tq, 'hqInput':X_test_hq})
Y_pred_final = np.empty(Y_pred.shape)
for i in range(len(Y_pred)):
	if(Y_pred[i] > 0.5):
		Y_pred_final[i] = 1
	else:
		Y_pred_final[i] = 0
"""print str(Y_pred[0:10])
print str(Y_test[0:10])"""
#score = model.evaluate(Y_test, Y_pred, verbose=0)
#score = accuracy_score(Y_test, Y_pred_final)
precision, recall, f1, support = precision_recall_fscore_support(Y_test, Y_pred_final, average='binary')
print('Precision: ' + str(precision) + '\tRecall: ' + str(recall) + '\tF1: ' + str(f1))
