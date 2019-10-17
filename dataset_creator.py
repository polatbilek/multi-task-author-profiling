import os
from bs4 import BeautifulSoup as bs
from requests import get  # to make GET request
from collections import OrderedDict
import sys
import nltk
import xml.etree.cElementTree as ET
import requests
import argparse

link_ak = "https://en.wikipedia.org/wiki/Wikipedia:Links_to_(disambiguation)_pages/A-K" #2-12 (2 and 12 included)
link_lz = "https://en.wikipedia.org/wiki/Wikipedia:Links_to_(disambiguation)_pages/L-Z" #0-14 (0 and 14 included)
link = link_ak

dataset_folder = "C:\\Users\\polat\\Desktop\\tobedeleted"#"/home/darg2/Desktop/dataset"
wikipedia = "https://en.wikipedia.org"

print("----- Data collection has started -----")

#parser = argparse.ArgumentParser()
#parser.add_argument('-i', type=int, help='Number of the char')

#args = parser.parse_args()

char_index = 2#args.i


flag = True

# try to get html page of collection of disambiguation pages
while(flag):
	try:
		obj = get(link)
		flag = False
	except:
		continue

soup = bs(obj.content, "lxml")

ambigous_links = []

# Parse the disambiguas links
for element in soup.find_all("ol")[char_index].find_all("li"):
	ambigous_link = element.find_all("a")[0]["href"]
	if "/wiki/" in ambigous_link:
		ambigous_links.append(ambigous_link)

ambigous_links = ambigous_links[3:-1]

for ambigous_link in ambigous_links:

	links_to_go = []
	specific_links = []

	target = ambigous_link.split("/wiki/")[1].split("_(disambiguation)")[0]
	absolute_link = wikipedia + ambigous_link

	flag = True
	while(flag):
		try:
			get_req_obj = get(absolute_link)
			flag = False
		except:
			continue

	soup = bs(get_req_obj.content, "lxml")

	for i in range(len(soup.find_all("a"))):
		try:
			specific_links.append(soup.find_all("a")[i]["href"])
		except:
			continue

	# get the links of disambiguas elements
	# we saw that ?php files are not directed to a disambiguation page so https not wanted
	not_wanted_words_in_link = ["https", "(disambiguation)", "(name)", "(given_name)", "(surname)"]


	for link in specific_links:
		safe_link = True
		if str("/wiki/"+target+"_(") in link:
			for not_wanted in not_wanted_words_in_link:
				if not_wanted in link:
					safe_link = False

			if safe_link == True:
				links_to_go.append(link)

	safe_links = list(OrderedDict.fromkeys(links_to_go))

	if len(safe_links) > 1:
		print(safe_links)

		root = ET.Element("root")
		entity_count = 1
		for link in safe_links:

			absolute_link = wikipedia + link
			final_form = ""
			text = ""
			prev = ""
			flag = True

			wiki_title = link.split("/wiki/")[1].replace("_"," ")


			while (flag):
				try:
					response = requests.get('https://en.wikipedia.org/w/api.php',
											params={
												'action': 'query',
												'format': 'json',
												'titles': wiki_title,
												'prop': 'extracts',
												'explaintext': True, }).json()

					page = next(iter(response['query']['pages'].values()))
					flag = False
				except:
					continue

			try:
				page_text = page['extract']
			except:
				continue


			not_wanted_subtopics = ["Discography", "Albums", "Track listing", "Music videos",
									"Chart performance", "Personnel", "Videoclip",
									"Live performances", "Releases", "Credits", "Literature",
									"People"]

			end_subtopics = ["See also", "References", "External links", "Sources", "Notes"]

			check = False

			####################################################
			#### Parse the informative texts of the page ###
			for line in page_text.split("\n"):

				if line == "":
					continue

				for not_wanted_subtopic in not_wanted_subtopics:
					if not_wanted_subtopic in line:
						gather = False

				for end_subtopic in end_subtopics:
					if end_subtopic in line:
						breaker = True

				if check == False and "==" in line:
					check = True
					continue

				if check == True:
					if "==" not in line:
						if len(line.strip().split(" ")) >= 15:
							text += line + "\n"

			## arrange target name that is splittable from the text

			try:
				target_name = target.replace("_", " ")
			except:
				target_name = target

			del_flag = False
			new_text = ""


			##############################################
			####  TAGGING THE GROUND TRUTH ####
			splitted_index = 0
			document = ""

			for split_text in text.split(target_name):
				if splitted_index+1 < len(text.split(target_name)):
					document += split_text + "[e" + str(entity_count) + "]"
				else:
					document += split_text

				splitted_index += 1

			document.replace("\n\n", "\n")

			if "[e" + str(entity_count) + "]" in document:
				if len(document.split(" ")) > 15:

					doc = ET.SubElement(root, "doc")
					doc.text = document
					entity_count += 1


		if entity_count > 2:
			tree = ET.ElementTree(root)
			save_location = os.path.join(dataset_folder, str(target+".xml"))
			tree.write(save_location, encoding="UTF-8")

