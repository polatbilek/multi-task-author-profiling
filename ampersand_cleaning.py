# This code is created to change escape characters of xml into xml defined format (e.g. "&" -> "&amp;")

import os
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer

path = '/home/darg2/Desktop/blogs'
target_path = '/home/darg2/Desktop/new_blogs'
tokenizer = TweetTokenizer()


for blog in tqdm(os.listdir(path)):
	count = 0
	try:
		with open(os.path.join(path,blog),"r", encoding="utf8") as f:
			new_f = open(os.path.join(target_path,blog), "w", encoding="utf8")
			for line in f:
				new_line = ""
				for word in tokenizer.tokenize(line):
					if word not in ["<Blog>", "</Blog>", "<date>", "</date>", "<post>", "</post>"]:
						w = ""
						for char in word:
							if char == "&":
								w += "&amp;"
							elif char == "<":
								w += "&lt;"
							elif char == ">":
								w += "&gt;"
							elif char == "\'":
								w += "&apos;"
							elif char == "\"":
								w += "&quot;"
							else:
								w += char
						word = str(w)
					new_line += " " + word
				new_f.write(new_line)
			new_f.close()
	except:
		count += 1 # to count how many files couldn't opened. There was 1 corrupted file for my dataset.
		print(count)



