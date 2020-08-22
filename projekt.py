# Imports:

import os
import re
import sys
import ssl
import nltk
import math
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer

# Constants:

NUM_ARGS = 1
LAN_ENGLISH = 1
LAN_GERMAN = 2
MINIMUM_LENGTH_TOKEN = 3

# Global variables:


# Functions:

def ensureInputRange(msg, min, max):
	userInput = int(input(msg))
	while userInput > max or userInput < min:
		userInput = int(input(msg))

	return userInput

def ensureString(msg):
	userInput = str(input(msg))
	while (len(userInput)) == 0:
		userInput = str(input(msg))
	return userInput

def showInitialMenu():
	print("*****")
	print("Welcome to my search engine")
	print("Mario Varona Bueno")
	print("*****")
	if (len(sys.argv) != (NUM_ARGS + 1)):
		print("Please, specify the directory where you want to search")
		print("Example:")
		print("projekt.py corpus")
		print("Quitting...")
		sys.exit(2)
	else:
		dirName = sys.argv[1]
		lan = ensureInputRange("Specify the language of your query (English = 1, German = 2): ", LAN_ENGLISH, LAN_GERMAN)
		print("Creating inverted index...")
		print("Please wait...")
		return dirName, lan

def getFilesFromAskedDir(dir):
	files = [f for f in os.listdir(dir)]
	exclude = ".DS_Store" # We exclude macOS config file
	filesCleaned = []

	for file in files:
		if file not in exclude:
			filesCleaned.append(file)
	
	return filesCleaned

def assignIDsToEachFile(files):
	i = 0
	ids = {}
	for file in files:
		ids[i] = file
		i = i + 1

	return ids

def initStemmer(lan):
	# We download here the 'punkt' module for NLTK, instead of asking the user to do on the install
	# Module ssl and the following try-else block is needed, as the url to download the 'punkt' library throws a SSL exception (caused by its domain certificate)

	try:
		_create_unverified_https_context = ssl._create_unverified_context
	except AttributeError:
		pass
	else:
		ssl._create_default_https_context = _create_unverified_https_context

	nltk.download('punkt')

	if (lan == LAN_ENGLISH):
		stemmer = SnowballStemmer("english")
	elif (lan == LAN_GERMAN):
		stemmer = SnowballStemmer("german")

	return stemmer

def stemSentence(sentence, stemmer):
	token_words = word_tokenize(sentence)
	stem_sentence = []
	for word in token_words:
		if (len(word) > MINIMUM_LENGTH_TOKEN):
			if word not in stem_sentence:
				stem_sentence.append(stemmer.stem(word))
	
	return stem_sentence

def createTokenListForFile(fileName, dirName, stemmer):

	file = open(dirName + os.sep + fileName)
	my_lines_list = [line.lower() for line in file.readlines() if line.strip()]
	tokenizedStemmedFile = []
	i = 0
	for line in my_lines_list:
		tokenizedStemmedFile.extend(stemSentence(my_lines_list[i], stemmer))
		i = i + 1

	return tokenizedStemmedFile

def createTokenListForFiles(files, dirName, stemmer, ids):
	tokenList = {}
	i = 0
	for file in files:
		tokenizedFile = list(set(createTokenListForFile(file, dirName, stemmer)))

		for word in tokenizedFile:
			if not tokenList.get(word):
				tokenList[word] = []

			tokenList[word].append(list(ids.keys())[i])

		i = i + 1
	
	return tokenList

def listFrequencyForFile(dirName, file, isQuery):
	words = []
	if (isQuery == True):
		words = re.findall(r'\w+', file)
	else:
		with open(dirName + os.sep + file) as f:
		    passage = f.read().lower()
		words = re.findall(r'\w+', passage)

	words_lower = [word for word in words if (len(word) > MINIMUM_LENGTH_TOKEN)]
	frequency = Counter(words_lower)
	return frequency

def createSearchTerms(stemmer):
	terms = ensureString("Specify your search terms with a space ( ) as separator: ").lower().split(" ")
	terms_stem = []
	for term in terms:
		terms_stem.append(stemmer.stem(term))
	return terms, terms_stem

def calculateTfForFile(term, dirName, file, isQuery):
	word_frequencies = listFrequencyForFile(dirName, file, isQuery)
	term_frequency = word_frequencies[term]
	max_frequency = max(list(word_frequencies.values()))
	tf = term_frequency / max_frequency
	return tf

def calculateTfs(relevance, tokenList, terms, terms_stem, dirName, ids):
	i = 0

	for term in terms:
		
		if (terms_stem[i] in tokenList):
			files_with_term = []
			files_with_term.append(tokenList[terms_stem[i]])

			for file_with_term in files_with_term:
				file = ids[file_with_term]
				isQuery = False
				if (file_with_term == max(list(ids.keys()))):
					isQuery = True
				relevance[file_with_term][terms_stem[i]] = calculateTfForFile(term, dirName, file, isQuery)

		i = i + 1

	return relevance

def calculateIdfs(tokenList, terms, terms_stem, ids):
	i = 0
	numDocs = len(ids)
	for term in terms:
		print("IDF for term '" + term + "':")

		if (terms_stem[i] in tokenList):
			files_with_term = tokenList[terms_stem[i]]
			num_files_with_term = len(files_with_term)
			idf = math.log(numDocs / num_files_with_term)
			print("\tidf = " + str(idf))
		else:
			print("\tThis term has not been included in the inverted index")

		i = i + 1

def addQueryDocument(ids, tokenList, terms, terms_stem):
	newKey = max(list(ids.keys())) + 1
	ids[newKey] = ("Query (" + ' '.join(terms) + ")")

	for term in terms_stem:
		if (term not in tokenList):
			tokenList[term] = []
		tokenList[term].append(newKey)

	return ids, tokenList

def createRelevanceMatrix(tokenList, ids):
	relevance = dict()
	for doc in ids:
		relevance[doc] = dict()
		for term in tokenList:
			relevance[doc][term] = 0
	return relevance

# Entry point:

dirName, lan = showInitialMenu()
files = getFilesFromAskedDir(dirName)
ids = assignIDsToEachFile(files)
stemmer = initStemmer(lan)
tokenList = createTokenListForFiles(files, dirName, stemmer, ids)
terms, terms_stem = createSearchTerms(stemmer)
#print(terms, terms_stem)
#print(tokenList)
ids, tokenList = addQueryDocument(ids, tokenList, terms, terms_stem)
#print(tokenList)
relevance = createRelevanceMatrix(tokenList, ids)
#tfs = calculateTfs(relevance, tokenList, terms, terms_stem, dirName, ids)
#idfs = calculateIdfs(relevance, tokenList, terms, terms_stem, ids)
print(relevance)