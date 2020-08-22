from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk
import ssl
import re
from collections import Counter

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

stemmer = SnowballStemmer("english")

def stemSentence(sentence):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
    	if (len(word) > 3):
	        stem_sentence.append(stemmer.stem(word))
    
    return stem_sentence

def createTokenListForFile(fileName):

	file = open(fileName)
	my_lines_list = [line.lower() for line in file.readlines() if line.strip()]
	tokenizedStemmedFile = []
	i = 0
	for line in my_lines_list:
		tokenizedStemmedFile.extend(stemSentence(my_lines_list[i]))
		i = i + 1

	return tokenizedStemmedFile

def listFrequencyForFile(file):
	with open(file) as f:
	    passage = f.read().lower()

	words = re.findall(r'\w+', passage)
	words_lower = [word for word in words if (len(word) > 3)]
	frequency = Counter(words_lower)
	return frequency

dict = listFrequencyForFile('corpus-en/test.txt')
print(dict["data"])