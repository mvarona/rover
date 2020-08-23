from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk
import ssl
import re
from collections import Counter
import numpy as np

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

stemmer = SnowballStemmer("english")

def stem_sentence(sentence):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
    	if (len(word) > 3):
	        stem_sentence.append(stemmer.stem(word))
    
    return stem_sentence

def create_token_list_for_file(file_name):

	file = open(file_name)
	my_lines_list = [line.lower() for line in file.readlines() if line.strip()]
	tokenized_stemmed_file = []
	i = 0
	for line in my_lines_list:
		tokenized_stemmed_file.extend(stem_sentence(my_lines_list[i]))
		i = i + 1

	return tokenized_stemmed_file

def list_frequency_for_file(file):
	with open(file) as f:
	    passage = f.read().lower()

	words = re.findall(r'\w+', passage)
	words_lower = [word for word in words if (len(word) > 3)]
	frequency = Counter(words_lower)
	return frequency

def multiply_vectors(a, b):
	c = a.dot(b)
	return c

l1 = {"a": 1, "b": 0, "c": 0, "d": 1, "e": 0, "f": 0, "g": 0, "h": 0}
l2 = {"a": 1, "b": 1, "c": 0, "d": 4, "e": 0, "f": 0, "g": 0, "h": 1}

a = np.array(list(l1.values()))
b = np.array(list(l2.values()))
c = multiply_vectors(a, b)
print(c)