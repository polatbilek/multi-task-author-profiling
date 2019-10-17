import numpy as np
from parameters import FLAGS
import xml.etree.ElementTree as xmlParser
from nltk.tokenize import TweetTokenizer
import os
import sys
import random
import io
from tqdm import tqdm


#########################################################################################################################
# Read FastText Embeddings
#
# input: String (path)        - Path of embeddings to read
#
# output: dict (vocab)        - Dictionary of embeddings as value words as key
def readFastTextEmbeddings(path):
	fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')

	embeddings = []
	vocab = {}
	i = 0
	fin.readline() #first line is metadata about embeddings

	for line in tqdm(fin):
		tokens = line.rstrip().split(' ')
		embeddings.append(np.asarray(list(map(float, tokens[1:]))))
		vocab[tokens[0]] = i
		i += 1

	vocab["<PAD>"] = i
	i  += 1
	embeddings.append(np.random.randn(FLAGS.embedding_size)) #Padding vector
	vocab["<UNK>"] = i
	embeddings.append(np.random.randn(FLAGS.embedding_size)) #Unknown word vector

	return np.array(embeddings), vocab


def trim_data(data):
	new_data=[]
	
	for i in range(len(data)):
		if data[i][0] < 10000:
			new_data.append(data[i])

	return new_data


def max_seq_len(data):

	max_len = 0
	for point in data:
		if point[0] > max_len:
			max_len = point[0]

	return max_len


#########################################################################################################################
# Reads training dataset
# one-hot vectors:
#
# Gender: female = [0,1], male   = [1,0]
#
# Age: There are 8 age group, 10-15,15-20,20-25 ..etc
#	   Length of each onehot is 8 and the index where the age falls into the group is 1 (e.g. 16-> [0,1,0,0,0,0,0,0])
#
# Job: There are 40 jobs, the index where the jobs name has seen is 1 others 0 (if it is 3rd unique job, 2nd index is 1)
#
# input:  string = path  to the training data
# output: ground_truth (dict): key=user_id, value = list [age, job, gender] each feature is a one-hot vector
#         data (list): each element has a list= [sequence_length, user_id, text(as a list of words)]
def readData(path):
	tokenizer = TweetTokenizer()
	job_index = 0

	ground_truth = {}
	jobs = {}
	data = []

	for user in tqdm(os.listdir(path)):
		infos = user.split(".")
		features = []

		# preparing age info of author
		one_hot_age = np.zeros(8)
		one_hot_age[int((int(infos[2])-10)/5)] = 1

		features.append(one_hot_age)

		# preparing job info of author
		if infos[3] not in jobs:
			jobs[infos[3]] = job_index
			job_index += 1

		one_hot_job = np.zeros(40)
		one_hot_job[jobs[infos[3]]] = 1

		features.append(one_hot_job)

		# preparing gender info of author
		if infos[1] == "male":
			features.append([1, 0])
		else:
			features.append([0, 1])


		ground_truth[infos[0]] = features # saving the author info to ground truth dict

		xml_file_name = os.path.join(path, user)
		xmlFile = open(xml_file_name, "r", encoding="utf8")

		# Here we read the xml files, however there are 670 corrupted files, we just ignore them and don't read.
		try:
			for post in xmlParser.parse(xmlFile).findall("post"):
				words = tokenizer.tokenize(post.text)
				data.append([len(words), infos[0], words])
		except:
			pass

	data = trim_data(data)

	random.shuffle(data)

	return ground_truth, data


#########################################################################################################################
# Changes tokenized words to their corresponding ids in vocabulary
#
# input: list (data)   - List of data
#        dict (vocab)  - Dictionary of the vocabulary
#
# output: list (batch) - List of corresponding ids of words in the text w.r.t. vocabulary
def word2id(data, vocab):
	batch = []

	for i in range(len(data)): #loop of users
		data_as_wordids = []

		for word in data[i]: #loop in words of each text
			if word != "<PAD>":
				word = word.lower()

			try:
				data_as_wordids.append(vocab[word])
			except:
				data_as_wordids.append(vocab["<UNK>"])

		batch.append(data_as_wordids)

	return batch



#########################################################################################################################
# Returns the age, job, gender info of the users of the batch
#
# input: list (batch_users)   - List of users in the batch
#        dict (ground_truth)  - Ground truth info of all users
#
# output: list (targets)      - List of target values of users in the batch
def user2target(batch_users, ground_truth):
	targets = []

	for user in batch_users:
		targets.append(ground_truth[user])

	return targets


#########################################################################################################################
# Prepares batch data, also adds padding to texts
#
# input: list (data)         - List of texts corresponding to the users
#	     list (users)        - List of users
#	     dict (ground_truth) - Ground-truth info of each user
#	     dict (vocabulary)   - Vocabulary with words as key, id as value
#	     int  (iter_no)      - Current # of iteration we are on
#
# output: list (batch_input)       - Ids of each words to be used in tf_embedding_lookup
# 	      list (targets_age)       - Target values for age ground turth to be fed to the rnn
#		  list (targets_job)       - Target values for job ground turth to be fed to the rnn
#		  list (targets_gender)    - Target values for gender ground turth to be fed to the rnn
#	      list (batch_seqlen)      - Number of words in each text (gives us the # of time unrolls)
def prepWordBatchData(data, users, ground_truth, vocabulary, iter_no):

	start = iter_no * FLAGS.batch_size
	end = iter_no * FLAGS.batch_size + FLAGS.batch_size

	if end > len(data):
		end = len(data)

	batch_data = data[start:end]
	batch_users = [d[1] for d in batch_data]
	batch_seqlen = [d[0] for d in batch_data]

	batch_targets = user2target(batch_users, ground_truth)

	# prepare input by adding padding
	max_text_length = max(batch_seqlen)

	batch_input = []

	for text in batch_data:

		padded_text = []
		for j in range(max_text_length):
			if len(text[2]) > j:
				padded_text.append(text[2][j])
			else:
				padded_text.append("<PAD>")
		batch_input.append(padded_text)


	batch_input = word2id(batch_input, vocabulary)


	#user level shuffling
	c = list(zip(batch_input, batch_targets, batch_seqlen))
	random.shuffle(c)
	batch_input, batch_targets, batch_seqlen = zip(*c)

	targets_age = [t[0] for t in batch_targets]
	targets_job = [t[1] for t in batch_targets]
	targets_gender = [t[2] for t in batch_targets]

	return batch_input, targets_age, targets_job, targets_gender, batch_seqlen




#########################################################################################################################
# partites the data into 3 part training, validation, test
#
# input: list (data)           - List of data and info about data (e.g. [seqlen, userid, text])
#	     dict (ground_truth)   - Ground truth info of authors (key=userid, value=[age, job, gender])
#
# output: output_format (list) : "*_data" same format but the "*_users" are just userid.
# 						         To get user info, you need to query ground-truth dictionary like gt['11213'].
def partite_dataset(data, ground_truth):

	training_set_size = int(len(list(ground_truth.keys())) * FLAGS.training_set_size)
	valid_set_size = int(len(list(ground_truth.keys())) * FLAGS.validation_set_size) + training_set_size

	training_data = []
	valid_data = []
	test_data = []

	users = list(ground_truth.keys())
	training_users = users[:training_set_size]
	valid_users = users[training_set_size:valid_set_size]
	test_users = users[valid_set_size:]

	for post in data:
		if post[1] in training_users:
			training_data.append(post)

		elif post[1] in valid_users:
			valid_data.append(post)

		elif post[1] in test_users:
			test_data.append(post)


	print("Training data-set size=" + str(len(training_data)) + " Validation data-set size=" + str(len(valid_data)) + " Test data-set size=" + str(len(test_data)))
	print("Training user-set size=" + str(len(training_users)) + " Validation user-set size=" + str(len(valid_users)) + " Test user-set size=" + str(len(test_users)))

	return training_data, training_users, valid_data, valid_users, test_data, test_users

