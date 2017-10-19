import gensim
import nltk
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
       fname = open(self.filename,'r')
       for line in fname:
                line_mod = line.strip().split('||')
		words1 = line_mod[0].split(" ")
		words2 = line_mod[1].split(" ")
		words_mod1 = []
		words_mod2 = []
		for word in words1:
			if word!='':
				words_mod1.append(word)
		yield words_mod1
		for word in words2:
			if word!='':
				words_mod2.append(word)
		yield words_mod2
 
sentences = MySentences("test.txt") # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences,min_count=1)
inp_file = open("test.txt",'r')
out_file = open("out2.txt",'w')
for line in inp_file:
	sen1 = line.strip().split('||')[0]
	sen2 = line.strip().split('||')[1]
	words1 = sen1.split(" ")
	words2 = sen2.split(" ")
	words_mod1 = []
	words_mod2 = []
	for word in words1:
		if word!='':
			words_mod1.append(word)
	for word in words2:
		if word!='':
			words_mod2.append(word)
	score = 0.00
	for word1 in words_mod1:
		for word2 in words_mod2:	
			value = model.similarity(word1,word2)
			if value > 0:
				score = value + score
	if score > 0.5:
		out_file.write(sen1 + '||' + sen2 + '||' + str(score) + '\n')
		
