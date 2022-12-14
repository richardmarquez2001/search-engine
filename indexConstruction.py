# from collections import OrderedDict
from bs4 import BeautifulSoup as bs
import porterAlgo  # Porter Stemming Algo

# import json
import gzip
import regex as re
import time
import orjson
# import sys
from tqdm import tqdm
# import lxml
import os

from helperFiles import helpermaker


class index_construction:

    split_postings = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

    def __init__(
        self,
        file_path,
        stopwords_path,
        stem_on,
        stopwords_on,
    ):
        self.file_path = file_path
        self.stopwords_path = stopwords_path
        self.stem_on = stem_on
        self.stopwords_on = stopwords_on
        self.extension = self.get_extension()
        self.stop_words = []
        self.document_title_dict = {}
        self.document_frequency = {}
        self.postinglist_dict = {}
        self.clean_docs_dict = {}
        self.clean_docs_dict_true = {}
        self.stemmed_cache = {}
        self.exec_time = 0
        self.title_file_name = "docTitleDict"
        self.posting_file_name = "postingListDict"
        self.df_file_name = "documentFrequency"
        self.clean_docs_file_name = "cleanDocsDict"
        self.names = ["a-e", "f-j", "k-o",
                      "p-t", "u-z", "num"]

    def get_extension(self):
        # Get correct File Path Extension
        if self.stem_on:
            extension = "_stem"
        else:
            extension = "_nostem"
        if self.stopwords_on:
            extension += "_stop"
        else:
            extension += "_nostop"
        return extension

    def init_path(self, path):
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb+') as f:
                f.write(orjson.dumps({}))
        else:
            with open(path, 'wb+') as f:
                f.write(orjson.dumps({}))

    def make_dir(self):
        path = self.extension[1:]
        print(os.path.exists(path))
        if not os.path.exists(path):
            os.makedirs(path)
            print("ok")
        # self.init_path(path)

    def make_new_files(self):
        # self.make_dir()
        for name in self.names:
            self.init_path(
                self.extension[1:]+"/"+self.posting_file_name+self.extension+"/"+name+".json")
        # make cleaned doc files
        self.init_path(
            self.extension[1:]+"/"+self.clean_docs_file_name+self.extension+".json")
        self.init_path(
            self.extension[1:]+"/"+self.clean_docs_file_name+self.extension+"_true"+".json")

    def get_stopwords(self):
        if not self.stopwords_on:
            with open(self.stopwords_path, 'r') as stopword:
                self.stop_words = [word.replace("\n", "") for word in stopword]

    def porter_stemming(self, uncleaned_word):
        start_time = time.time()
        porterstemming_algo = porterAlgo.PorterStemmer()

        if uncleaned_word.isnumeric():
            return uncleaned_word  # keep numbers as is

        word = re.sub('[^a-zA-Z]', '', uncleaned_word)  # remove special chars

        # remove stopwords
        if word.lower().replace("\n", "") in self.stop_words or not len(word) > 2:
            return 'None'

        if self.stem_on == True:
            if word not in self.stemmed_cache:
                word_cleaned = porterstemming_algo.stem(word, 0, len(
                    word)-1).lower().replace("\n", "")  # lemmatize token
                self.stemmed_cache[word] = word_cleaned
            else:
                word_cleaned = self.stemmed_cache[word]
        else:
            word_cleaned = word.lower().replace("\n", "")
        return word_cleaned

    def remove_none(self, word):
        bad = ['None']
        if word in bad:
            return False
        return True

    def doc_frequency(self, doc_set):
        # Get doc frequency
        for word in doc_set:
            self.document_frequency[word] = 1 + \
                self.document_frequency.get(word, 0)

    def add_to_post(self):
        for i in range(len(self.names)):
            with open(self.extension[1:]+"/"+self.posting_file_name+self.extension+"/"+self.names[i]+".json", "rb+") as postings:
                self.postinglist_dict = orjson.loads(postings.read())
            for doc_id, index, term in self.split_postings[i]:
                if term in self.postinglist_dict:
                    if doc_id in self.postinglist_dict[term]:
                        self.postinglist_dict[term][doc_id][0] += 1
                        self.postinglist_dict[term][doc_id][1] += f',{index}'
                    else:
                        self.postinglist_dict[term][doc_id] = [
                            1, f'{index}']
                else:
                    self.postinglist_dict[term] = {
                        str(doc_id): [1, f'{index}']}
            with open(self.extension[1:]+"/"+self.posting_file_name+self.extension+"/"+self.names[i]+".json", 'wb') as f:
                f.write(orjson.dumps(self.postinglist_dict))
            del self.postinglist_dict
            self.postinglist_dict = {}
        # del self.split_postings
        self.split_postings = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

    def postings_list(self, doc_id, doc):
        seen = set()
        for index, term in enumerate(doc):
            if term not in seen:
                self.document_frequency[term] = 1 + \
                    self.document_frequency.get(term, 0)
                seen.add(term)
            match ord(term[0]):
                case letter if ord("a") <= letter <= ord("e"):
                    file = 0

                case letter if ord("f") <= letter <= ord("j"):
                    file = 1

                case letter if ord("k") <= letter <= ord("o"):
                    file = 2

                case letter if ord("p") <= letter <= ord("t"):
                    file = 3

                case letter if ord("u") <= letter <= ord("z"):
                    file = 4
                case _:
                    file = 5
            self.split_postings[file].append((doc_id, index, term))

    def process_contents(self, line):
        # Load JSON
        data = orjson.loads(line)
        doc_id = str(data['id'])
        title = data['title']

        # Save doc title
        self.document_title_dict[doc_id] = title
        # Deal with HTML content

        contents = bs(data['contents'], "lxml")

        # # Remove HTML tags from contents
        contents_stripped = " ".join(contents.stripped_strings)
        contents_stripped = contents_stripped.split(" ")

        # Stem contents
        # Stem on
        porter_stem = map(self.porter_stemming, contents_stripped)
        porter_stem = list(filter(self.remove_none, porter_stem))

        # Stem off
        self.stem_on = False
        contents_stripped = map(self.porter_stemming, contents_stripped)
        contents_stripped = list(filter(self.remove_none, contents_stripped))
        self.stem_on = True

        return [doc_id, porter_stem, contents_stripped]

    def process_files(self):
        count = 0
        with gzip.open(self.file_path, 'rb') as file:
            for wiki in tqdm(file):
                count += 1

                # Parse wiki doc and get contents
                doc_id, contents, pre_contents = self.process_contents(wiki)

                # Update Positing List
                self.postings_list(doc_id, contents)

                # Save cleaned
                self.clean_docs_dict[doc_id] = " ".join(
                    pre_contents)  # Unstemmed dict
                self.clean_docs_dict_true[doc_id] = " ".join(
                    pre_contents)  # Stemmmed dict

                if count % 10000 == 0:
                    self.save()
                if count == 10000:
                    break
            self.save()
            helpermaker()

    def save_df(self):
        with open(self.extension[1:]+"/"+self.df_file_name+self.extension+".json", 'wb') as f:
            f.write(orjson.dumps(self.document_frequency))

    def save_cd(self):
        with open(self.extension[1:]+"/"+self.clean_docs_file_name+self.extension+".json", 'wb') as f:
            f.write(orjson.dumps(self.clean_docs_dict))

        with open(self.extension[1:]+"/"+self.clean_docs_file_name+self.extension+"_true"+".json", 'wb') as f:
            f.write(orjson.dumps(self.clean_docs_dict_true))

    def save_dt(self):
        with open(self.extension[1:]+"/"+self.title_file_name+self.extension+".json", 'wb') as f:
            f.write(orjson.dumps(self.document_title_dict))

    # Unstemmed
    def load_cd(self):
        with open(self.extension[1:]+"/"+self.clean_docs_file_name+self.extension+".json", 'rb') as clean:
            return orjson.loads(clean.read())

    # Stemmed
    def load_cd_true(self):
        with open(self.extension[1:]+"/"+self.clean_docs_file_name+self.extension+"_true"+".json", 'rb') as clean:
            return orjson.loads(clean.read())

    def load_and_merge_cd(self):
        self.clean_docs_dict = self.load_cd() | self.clean_docs_dict
        self.clean_docs_dict_true = self.load_cd_true() | self.clean_docs_dict_true
        self.save_cd()
        del self.clean_docs_dict
        del self.clean_docs_dict_true
        self.clean_docs_dict = {}
        self.clean_docs_dict_true = {}

    def save(self):
        self.add_to_post()
        self.load_and_merge_cd()
        self.save_df()
        self.save_dt()


# IC = index_construction(
#     r"/Users/nojiro/Desktop/CPS842/trec_corpus_5000.jsonl.gz", "cacm_stopwords.txt", True, False)
IC = index_construction(
    r"C:\Users\jkyle\Desktop\CPS842\trec_corpus_5000.jsonl.gz", "cacm_stopwords.txt", True, False)

IC.get_stopwords()
IC.make_new_files()
IC.process_files()
