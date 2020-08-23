# Imports:

import os
import re
import sys
import ssl
import nltk
import math
import numpy as np
from collections import Counter
from operator import itemgetter
from collections import OrderedDict
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# Constants:

NUM_ARGS = 1
LAN_ENGLISH = 1
LAN_GERMAN = 2
MINIMUM_LENGTH_TOKEN = 3
MAX_DECIMAL_FLOAT = 4

# Functions:

def ensure_input_range(msg, min, max):
	user_input = int(input(msg))
	while user_input > max or user_input < min:
		user_input = int(input(msg))

	return user_input

def ensure_string(msg):
	user_input = str(input(msg))
	while (len(user_input)) == 0:
		user_input = str(input(msg))
	return user_input

def show_initial_menu():
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
		dir_name = sys.argv[1]
		lan = ensure_input_range("Specify the language of your query (English = 1, German = 2): ", LAN_ENGLISH, LAN_GERMAN)
		print("Creating inverted index...")
		print("Please wait...")
		return dir_name, lan

def get_files_from_dir(dir):
	files = [f for f in os.listdir(dir)]
	files_cleaned = []

	for file in files:
		if file.endswith(".txt"):
			files_cleaned.append(file)
	
	return files_cleaned

def assign_ids_to_each_file(files):
	i = 0
	ids = {}
	for file in files:
		ids[i] = file
		i = i + 1

	return ids

def init_stemmer(lan):
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

def stem_sentence(sentence, stemmer):
	token_words = word_tokenize(sentence)
	stem_sentence = []
	for word in token_words:
		if (len(word) > MINIMUM_LENGTH_TOKEN):
			if word not in stem_sentence:
				stem_sentence.append(stemmer.stem(word))
	
	return stem_sentence

def create_token_list_for_file(file_name, dir_name, stemmer):

	file = open(dir_name + os.sep + file_name)
	my_lines_list = [line.lower() for line in file.readlines() if line.strip()]
	tokenized_stemmed_file = []
	i = 0
	for line in my_lines_list:
		tokenized_stemmed_file.extend(stem_sentence(my_lines_list[i], stemmer))
		i = i + 1

	return tokenized_stemmed_file

def create_token_list_for_files(files, dir_name, stemmer, ids):
	token_list = {}
	i = 0
	for file in files:
		tokenized_file = list(set(create_token_list_for_file(file, dir_name, stemmer)))

		for word in tokenized_file:
			if not token_list.get(word):
				token_list[word] = []

			token_list[word].append(list(ids.keys())[i])

		i = i + 1
	
	return token_list

def list_frequency_for_file(dir_name, file, is_query):
	words = []
	if (is_query == True):
		words = re.findall(r'\w+', file)
	else:
		with open(dir_name + os.sep + file) as f:
		    passage = f.read().lower()
		words = re.findall(r'\w+', passage)

	words_lower = [word for word in words if (len(word) > MINIMUM_LENGTH_TOKEN)]
	frequency = Counter(words_lower)
	return frequency

def create_search_terms(stemmer):
	terms = ensure_string("Specify your search terms with a space ( ) as separator: ").lower().split(" ")
	terms_stem = []
	for term in terms:
		terms_stem.append(stemmer.stem(term))
	return terms, terms_stem

def calculate_tf_for_file(term, dir_name, file, is_query):
	word_frequencies = list_frequency_for_file(dir_name, file, is_query)
	term_frequency = word_frequencies[term]
	max_frequency = max(list(word_frequencies.values()))
	tf = round(term_frequency / max_frequency, MAX_DECIMAL_FLOAT)
	return tf

def calculate_tfs(relevance, token_list, terms, terms_stem, dir_name, ids):
	i = 0

	for term in terms:
		
		if (terms_stem[i] in token_list):
			files_with_term = token_list[terms_stem[i]]

			for file_with_term in files_with_term:
				file = ids[file_with_term]
				is_query = False
				if (file_with_term == max(list(ids.keys()))):
					is_query = True
				relevance[file_with_term][terms_stem[i]] = calculate_tf_for_file(term, dir_name, file, is_query)

		i = i + 1

	return relevance

def calculate_idfs(relevance, token_list, terms, terms_stem, ids):
	i = 0
	num_docs = len(ids)
	for term in terms:
		if (terms_stem[i] in token_list):
			files_with_term = token_list[terms_stem[i]]
			num_files_with_term = len(files_with_term)
			idf = math.log(num_docs / num_files_with_term)
			for doc in range(0, num_docs):
				relevance[doc][terms_stem[i]] = round(relevance[doc][terms_stem[i]] * idf, MAX_DECIMAL_FLOAT)
		
		i = i + 1

	return relevance

def add_query_document(ids, token_list, terms, terms_stem):
	new_key = max(list(ids.keys())) + 1
	ids[new_key] = ("Query (" + ' '.join(terms) + ")")

	for term in terms_stem:
		if (term not in token_list):
			token_list[term] = []
		token_list[term].append(new_key)

	return ids, token_list

def create_relevance_matrix(token_list, ids):
	relevance = dict()
	for doc in ids:
		relevance[doc] = dict()
		for term in token_list:
			relevance[doc][term] = 0
	return relevance

def print_relevance(relevance):
	for doc in range(0, len(relevance)):
		print("Document " + str(doc) + ":")
		for term in relevance[0]:
			print(str(relevance[doc][term]) + "  ", end='')
		print("\n")

def magnitude(x):
	return round(math.sqrt(sum(i**2 for i in x.values())), MAX_DECIMAL_FLOAT)

def multiply_vectors(a, b):
	c = a.dot(b)
	return c

def create_similarity(relevance):
	similarity = {}
	query = len(relevance) - 1

	for doc in range(0, len(relevance) - 1):
		similarity[doc] = 0

	for doc in range(0, len(relevance) - 1):
		query_vector = np.array(list(relevance[query].values()))
		doc_vector = np.array(list(relevance[doc].values()))
		vector_product = multiply_vectors(query_vector, doc_vector)
		magnitude_product = magnitude(relevance[query]) * magnitude(relevance[doc])
		if (magnitude_product == 0):
			similarity[doc] = 0
		else:
			similarity[doc] = round(vector_product / magnitude_product, MAX_DECIMAL_FLOAT)

	return similarity

def order_similarity(similarity):
	ordered_similarity = OrderedDict(sorted(similarity.items(), key = itemgetter(1), reverse = True))
	return ordered_similarity

def print_similarity(ordered_similarity, ids):
	print("*** Most relevant files for query: ***")
	print("**** (According to its cosine similarity value) ****")
	print("")

	i = 0
	for ordered_result in ordered_similarity:
		print(str(i + 1) + ". File: " + ids[ordered_result] + ". Similarity with query: " + str(ordered_similarity[ordered_result]) + ". ID: " + str(ordered_result))
		i = i + 1

def print_contexts(ordered_similarity, dir_name, files, terms):
	print("*** Results with context: ***")
	print("")
	start_bold = "\033[1m"
	end_bold = "\033[0;0m"
	
	for ordered_result in ordered_similarity:
		if (ordered_similarity[ordered_result] > 0):
			print("File  " + files[ordered_result] + ":")
			file = open(dir_name + os.sep + files[ordered_result])		

			for term in terms:
				lines_with_occurrence = [line.lower() for line in file.readlines() if (line.strip() and term in line.lower())]
				i = 1
				for line in lines_with_occurrence:
					print("")
					print("\tOccurrence #" + str(i) + ":")
					print("")
					for word in line.split(' '):
						if (term in word):
							print(start_bold + word + end_bold + " ", end='')
						else:
							print(word + " ", end='')
					i = i + 1
	print("")

def show_end():
	print("")
	print("We hope we've been useful")
	print("Until next search!")
	print("")


# Entry point:

dir_name, lan = show_initial_menu()
files = get_files_from_dir(dir_name)
ids = assign_ids_to_each_file(files)
stemmer = init_stemmer(lan)
token_list = create_token_list_for_files(files, dir_name, stemmer, ids)
terms, terms_stem = create_search_terms(stemmer)
ids, token_list = add_query_document(ids, token_list, terms, terms_stem)
relevance = create_relevance_matrix(token_list, ids)
relevance = calculate_tfs(relevance, token_list, terms, terms_stem, dir_name, ids)
relevance = calculate_idfs(relevance, token_list, terms, terms_stem, ids)
similarity = create_similarity(relevance)
ordered_similarity = order_similarity(similarity)
print_similarity(ordered_similarity, ids)
print_contexts(ordered_similarity, dir_name, files, terms)
show_end()