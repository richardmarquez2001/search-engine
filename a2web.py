import porterAlgo

from io import BytesIO
import base64

import time
import orjson
import math
import regex as re
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)

from wordcloud import WordCloud

plt.switch_backend('Agg')
class BestMatch25:
    def __init__(
        self,
        query,
        relevant_doc_ids,
        postings,
        document_frequencys,
        corpus_info,
        k1=1.5,
        b=0.75
    ):
        self.query = query
        self.relevant_doc_ids = relevant_doc_ids
        self.postings = postings
        self.document_frequencys = document_frequencys
        self.corpus_size = corpus_info["doc_count"]
        self.average_doclen = corpus_info["total_length"]/self.corpus_size
        self.corpus_lengths = corpus_info["doc_lengths"]
        self.k1 = k1
        self.b = b
        self.idf = {}

    def calculate_idf(self, term):
        if term in self.document_frequencys:
            df = self.document_frequencys[term]
            self.idf[term] = math.log(
                1 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def load_idf(self):
        list(map(self.calculate_idf, self.query))

    def fit(self):
        self.load_idf()  # load up idf calculations
        scores = [self.score(doc_id) for doc_id in self.relevant_doc_ids]
        return scores

    def score(self, relevant_doc):
        result = 0.0
        doc_length = self.corpus_lengths[relevant_doc]
        for term in self.query:
            # check if term is in list of relevant docs
            if term in self.postings and relevant_doc in self.postings[term]:
                # term freq for doc
                freq = self.postings[term][relevant_doc][0]
                numerator = self.idf[term] * freq * (self.k1+1)
                denominator = freq + self.k1 * \
                    (1.0-self.b+self.b*doc_length/self.average_doclen)
                result += (numerator / denominator)
            else:
                continue
        return [relevant_doc, result]


class NaiveBayes:
    def __init__(self):
        with open(r"stem_nostop/NB_stem_nostop.json", "rb") as NB:
            self.prior, self.condprob, self.doc_topics, self.topics = orjson.loads(
                NB.read()).values()

    def getQueryTopic(self, terms):
        scores = {}
        topic_ids = self.topics.keys()
        for topic_id in topic_ids:
            if topic_id not in self.prior:
                continue
            scores[self.topics[topic_id]] = self.prior[topic_id]
            for term in terms:
                if term not in self.condprob:
                    continue
                scores[self.topics[topic_id]] += self.condprob[term][topic_id]
        return sorted(scores, key=scores.get, reverse=True)

    def getDocTopic(self, doc_id):
        out = []
        if doc_id in self.doc_topics:
            for topic in self.doc_topics[doc_id]:
                if topic in self.topics:
                    out.append(self.topics[topic])
            return out
        return None


class KNN:
    doc_topics = {}
    topics = {}

    def __init__(self):
        self.vectorizer = None
        self.clf = None

        with open(r"stem_nostop/topics_stem_nostop.json", "rb") as topics:
            self.topics = orjson.loads(topics.read())

    def fit(self):
        with open(r"stem_nostop/NB_stem_nostop.json", "rb") as NB:
            _, _, self.doc_topics, _ = orjson.loads(
                NB.read()).values()

        path = "stem_nostop/cleanDocsDict_stem_nostop_true.json"
        with open(path, 'rb') as f:
            cd = orjson.loads(f.read())

        X_train = []
        y_train = []
        for doc_id in self.doc_topics.keys():
            if doc_id in cd:
                topics = self.doc_topics[doc_id]
                for topic in topics:
                    X_train.append(cd[doc_id])
                    y_train.append(topic)
        self.vectorizer = TfidfVectorizer()
        vectors = self.vectorizer.fit_transform(X_train)
        knn = KNeighborsClassifier(n_neighbors=5)
        self.clf = knn.fit(vectors, y_train)

        knnFile = open('knn_model', 'wb')
        pickle.dump(self.clf, knnFile)
        knnFile.close()

        vectorFile = open('vector_model', 'wb')
        pickle.dump(self.vectorizer, vectorFile)
        knnFile.close()
        self.vectorizer = None
        self.clf = None

    def loadModel(self):
        self.clf = pickle.load(open('knn_model', 'rb'))
        self.vectorizer = pickle.load(open('vector_model', 'rb'))

    def predict(self, query):
        topics = self.clf.predict(self.vectorizer.transform(query))
        return self.topics[topics[0]]

class Query:
    postings = {}
    freqs = {}
    titles = {}
    cleaned_docs = {}
    corpus_info = {}
    topics_to_doc = {}
    stop_words = []

    def __init__(
        self
    ):
        self.stem_on = True
        self.stopwords_on = False
        self.stopwords_path = "cacm_stopwords.txt"
        self.title_file_name = "docTitleDict"
        self.posting_file_name = "postingListDict"
        self.df_file_name = "documentFrequency"
        self.clean_docs_file_name = "cleanDocsDict"
        self.topics_to_doc_file_name = "topics_to_doc"
        self.ci_file_name = "size"
        self.names = ["a-e", "f-j", "k-o",
                      "p-t", "u-z", "num"]
        self.porter_stemming_algo = porterAlgo.PorterStemmer()

        self.load_df()
        self.load_dt()
        self.load_ci()
        self.load_cd()
        self.load_ttd()

    def get_stopwords(self):
        with open(self.stopwords_path, 'r') as stopword:
            self.stop_words = [word.replace("\n", "") for word in stopword]

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

    def char_to_index(self, letter):
        match ord(letter):
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
        return file

    def load_postings_range(self, letter):
        extension = self.get_extension()
        self.postings = {}
        i = self.char_to_index(letter)
        with open(extension[1:]+"/"+self.posting_file_name+extension+"/"+self.names[i]+".json", "rb+") as postings:
            self.postings = orjson.loads(postings.read())

    def load_cd(self):
        extension = self.get_extension()
        if not self.cleaned_docs:
            with open(extension[1:]+"/"+self.clean_docs_file_name+extension+".json", 'rb') as clean:
                self.cleaned_docs = orjson.loads(clean.read())

    def load_df(self):
        extension = self.get_extension()
        if not self.freqs:
            with open(extension[1:]+"/"+self.df_file_name+extension+".json", 'rb') as freqs:
                self.freqs = orjson.loads(freqs.read())

    def load_dt(self):
        extension = self.get_extension()
        if not self.titles:
            with open(extension[1:]+"/"+self.title_file_name+extension+".json", 'rb') as titles:
                self.titles = orjson.loads(titles.read())

    def load_ci(self):
        extension = self.get_extension()
        if not self.corpus_info:
            print(extension[1:]+"/"+self.ci_file_name+extension+".json")
            with open(extension[1:]+"/"+self.ci_file_name+extension+".json", 'rb') as corpus_info:
                self.corpus_info = orjson.loads(corpus_info.read())

    def load_ttd(self):
        extension = self.get_extension()
        if not self.topics_to_doc:
            with open(extension[1:]+"/"+self.topics_to_doc_file_name+extension+".json", 'rb') as topics_to_doc:
                self.topics_to_doc = orjson.loads(topics_to_doc.read())

    def get_window(self, size, index, doc):
        # Gets size terms to left and right of index
        if index < size:
            window = doc[:index + size + 1]
        else:
            window = doc[index-size:index + size + 1]
        return " ".join(window)

    def remove_invalid(self, term):
        bad = ['', "None"]
        if term in bad or len(term) == 0:
            return False
        return True

    def porter_stemming(self, uncleaned_word):
        # Keep numbers as is
        if uncleaned_word.isnumeric():
            return uncleaned_word

        # Remove special chars
        word = re.sub('[^a-zA-Z]', '', uncleaned_word)

        # Remove stopwords
        if word.lower().replace("\n", "") in self.stop_words or not len(word) > 2:
            return 'None'

        # Stem
        word_cleaned = self.porter_stemming_algo.stem(word, 0, len(
            word)-1).lower().replace("\n", "")
        return word_cleaned

    def preprocess_query(self, query):
        # Split query and save un stemmed version for generating reult summary later.
        query_stem = query.split(" ")
        query_nostem = query.split(" ")

        for index, term in enumerate(query_stem):
            # Stem
            query_stem[index] = self.porter_stemming(term)
            # Still lower and replace for un stemmed version
            query_nostem[index] = term.lower().replace("\n", "")

        # Stem the query
        query_stem = map(self.porter_stemming, query_stem)
        query_stem = list(filter(self.remove_invalid, query_stem))
        # Remove duplicates
        query_stem = [*set(query_stem)]
        # query_nostem = list(filter(self.remove_invalid, query_nostem))

        # Split the query into posting list groups to easily load them later
        query_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        for term in query_stem:
            query_dict[self.char_to_index(term[0])].append(term)

        return query_stem, query_nostem, query_dict

    def idf(self, query):
        idf = {}
        for term in query:
            idf[term] = 1 + \
                math.log(
                    (1+self.corpus_info["doc_count"])/(1+self.freqs.get(term, 0)))
        return idf

    def tfidf(self, query, idf, doc_id):
        score = 0.0
        for term in query:
            # if term not in tf:
            tf = int(self.postings.get(term, {}).get(doc_id, [0])[0])
            score += tf * idf[term]
        return score/len(query)

    def get_topic_scores(self, query, topic):
        # Get docs with this category
        rel_docs = self.topics_to_doc[topic]
        scores_naive = []
        idf = self.idf(query)
        max_tfidf = 0
        for doc_id in rel_docs:
            tfidf = self.tfidf(query, idf, doc_id)
            max_tfidf = max(tfidf, max_tfidf)
            scores_naive.append([doc_id, tfidf])
        if max_tfidf != 0:
            scores_naive.sort(key=lambda scores: scores[1], reverse=True)
        scores = scores_naive
        return scores, max_tfidf

    def search_index(self, query, model):
        relevant_docs = {}
        relevant_postings = {}
        # Parse Query
        # Stem query if needed
        query, query_nostem, query_dict = self.preprocess_query(query)
        for range in query_dict.keys():
            if query_dict[range]:
                # load postings range
                self.load_postings_range(query_dict[range][0][0])
            for term in query_dict[range]:
                # Get doc frequency
                doc_freq = self.freqs.get(term, 0)
                # Get docs where query term is found. Returns an empty ditc if not
                relevant_postings[term] = self.postings.get(term, {})
                relevant_docs = relevant_docs | relevant_postings[term].keys()

        # Set postings to relavent postings and free memory
        self.postings = relevant_postings
        del relevant_postings

        # BM25
        if model == "bm25" or model == "all":
            bm25 = BestMatch25(query, relevant_docs,
                               self.postings, self.freqs, self.corpus_info)
            scores_bm = bm25.fit()
            del bm25
            scores_bm.sort(key=lambda scores: scores[1], reverse=True)
            scores = scores_bm

        # Naive Bayes
        nb = NaiveBayes()
        if model == "naive" or model == "all":
            # Generate the naive basyes classification of the query
            topics_naive = nb.getQueryTopic(query)

            if model == "naive":
                # Get docs with this category
                scores, max_score = self.get_topic_scores(
                    query, topics_naive[0])

        # KNN
        knn = KNN()
        knn.loadModel()
        if model == "knn" or model == "all":
            # Generate the knn classification of the query
            topic_knn = knn.predict([" ".join(query)])

            if model == "knn":
                # Get docs with this category
                scores, max_score = self.get_topic_scores(query, topic_knn)

        # All
        # Use rel docs from
        if model == "all":
            scores = []
            docs_by_topics = {}
            for doc_id, score in scores_bm:
                doc_topic = nb.getDocTopic(doc_id)[0]
                if doc_topic == topic_knn:
                    multi = 1.3
                else:
                    multi = 1
                if doc_topic in docs_by_topics:
                    docs_by_topics[doc_topic].append([doc_id, score*multi])
                else:
                    docs_by_topics[doc_topic] = [[doc_id, score*multi]]
            for topic in topics_naive:
                for d in docs_by_topics.get(topic, []):
                    scores.append(d)

        # Make results
        count = 0
        results = []
        for doc_id, score in scores[:10000]:
            count += 1
            doc_topic = nb.getDocTopic(doc_id)
            results.append({"rank": count, "doc_id": doc_id,
                            "title": self.titles[doc_id], "score": score, "topic": ", ".join(doc_topic), "summary": self.generate_summary(doc_id, query_nostem, 25)})

        visualization_data = self.query_visualization(results, query_nostem)
        return results, visualization_data
    
    def _convertToHTML(self, obj):
        canvas = FigureCanvasAgg(obj)
        png_output = BytesIO()
        canvas.print_png(png_output)

        # Encode the image data as a base64 string
        image_data = base64.b64encode(png_output.getvalue()).decode()

        return image_data
    
    # Visualization 
    def query_visualization(self, result, query_nostem):
        documents = []
        labels = []
        for doc in result[:30]: # Creates visualization for the top 30 results
            doc_id = doc["doc_id"]
            labels.append(doc["title"])
            documents.append(self.generate_summary(doc_id, query_nostem, 100))
        
        # Computing document similarities
        self.vectorizer = TfidfVectorizer()
        vectors = self.vectorizer.fit_transform(documents)
        terms = self.vectorizer.get_feature_names_out()

        doc_similarities = cosine_similarity(vectors)
        distance = 1 - doc_similarities
        clustering_matrix = ward(distance)
        
        # Dendrogram
        dendogram_i = plt.figure(figsize = (15,20))
        dendrogram(clustering_matrix, orientation="right", labels=labels)
        plt.tick_params(axis= 'x', which='both', bottom='off', top='off', labelbottom="on")
        plt.tight_layout() 
        dendogram_html = self._convertToHTML(dendogram_i)
        
        # Heatmap
        heatmap_i = plt.figure(figsize = (20,15))
        sn.heatmap(doc_similarities, xticklabels=labels, yticklabels=labels)
        plt.tight_layout() 
        heatmap_html = self._convertToHTML(heatmap_i)

        # Wordcloud
        bag_of_words = " ".join(terms)
        wordcloud = WordCloud(width = 850, height = 850,
                background_color ='white',
                min_font_size = 10).generate(bag_of_words)
        wordcloud_i = plt.figure(figsize = (15, 15), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 5)
        
        wordcloud_html = self._convertToHTML(wordcloud_i)
        
        return [dendogram_html,heatmap_html,wordcloud_html]
        
    def global_visualization(self, topics_to_doc, cleaned_docs):
        
        # Global Document count of topics bar graph
        topics = []
        counts = []
        for key, value in topics_to_doc.items():
            topics.append(key)
            counts.append(len(value))

        fig1 = plt.figure(figsize = (15, 15))
        plt.barh(topics, counts)
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=8)
        plt.xlabel("Topics", fontsize=10)
        plt.ylabel("Number of documents", fontsize=10)
        plt.title("Topics by the number of documents", fontsize=10)
        plt.tight_layout()
                
        # Global WordCloud
        bag_of_words = ""
        for key, value in cleaned_docs.items():
            bag_of_words += value + " "
            
        wordcloud = WordCloud(width = 850, height = 850,
                background_color ='white',
                min_font_size = 10).generate(bag_of_words)
 
        fig2 = plt.figure(figsize = (15, 15), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 5)
        
        fig1_html = self._convertToHTML(fig1)
        fig2_html = self._convertToHTML(fig2)
        
        return [fig1_html, fig2_html]
            
    def generate_summary(self, doc_id, query, size):
        summary = ""
        doc = self.cleaned_docs[doc_id].split(" ")
        query_index = []
        for term in query:
            if term in doc:
                query_index.append(doc.index(term))
        for index in query_index:
            summary += self.get_window(size, index, doc) + " ... "
        if summary == "":
            summary += self.get_window(size, size+1, doc) + " ... "
        return summary

    def search(self, query, model):
        # Handle queries
        start_time = time.time()
        results, qv = self.search_index(query, model)
        exec_time = time.time() - start_time

        return {"result": results, "time": exec_time, "query_visualizations": qv}
   
# Q = Query()
# Q.search_index("computer science", "all")
# Q.global_visualization(Q.topics_to_doc, Q.cleaned_docs)