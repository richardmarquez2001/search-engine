import csv
import orjson
import math
from tqdm import tqdm

import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import orjson
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


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


def helpermaker():
    path = "stem_nostop/cleanDocsDict_stem_nostop.json"
    t_len = 0
    count = 0
    doc_size = {}
    with open(path, 'rb') as f:
        cd = orjson.loads(f.read())
        for key in tqdm(cd.keys()):
            count += 1
            d_len = len(cd[key].split(" "))
            t_len += d_len
            doc_size[key] = d_len

    path = "stem_nostop/size_stem_nostop.json"
    with open(path, 'wb') as f:
        f.write(orjson.dumps({"total_length": t_len,
                "doc_count": count, "doc_lengths": doc_size}))

    # Make topics file
    path = "train_topics_keywords.tsv"
    keywords = {}
    tp = {}
    with open(path, encoding="utf-8") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            keywords[line[1]] = line[2].split(",")
            tp[line[0]] = line[1]
    # print(keywords)

    path = "stem_nostop\\topics_stem_nostop.json"
    with open(path, 'wb') as f:
        f.write(orjson.dumps(tp))

    # Convert tsv to doc_id --> topic dict
    path = "train_topics_reldocs.tsv"
    keywords = {}
    total = set()
    doc_to_topics = {}
    with open(path) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            # print(line)
            # print(line[1])
            keywords[line[1]] = line[2].split(",")
            for id in line[2].split(","):
                if id in doc_to_topics:
                    doc_to_topics[id].append(line[0])
                else:
                    doc_to_topics[id] = [line[0]]
                total.add(id)

    # Load stemmed dict
    path = "stem_nostop/cleanDocsDict_stem_nostop_true.json"
    with open(path, 'rb') as f:
        cd_keys = orjson.loads(f.read()).keys()

    # Convert tsv to topic --> doc_id[] dict
    path = "train_topics_reldocs.tsv"
    keywords = {}
    total = set()
    topics_to_doc = {}
    with open(path) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            all = line[2].split(",")
            rel = []
            for doc in all:
                if doc in cd_keys:
                    rel.append(doc)
            topics_to_doc[line[1]] = rel

    # Save Topics to docs
    path = "stem_nostop/topics_to_doc_stem_nostop.json"
    with open(path, 'wb') as f:
        f.write(orjson.dumps(topics_to_doc))

    path = "stem_nostop/cleanDocsDict_stem_nostop_true.json"
    with open(path, 'rb') as f:
        cd = orjson.loads(f.read())

    # set of words in the
    vocab = set()

    # total doc count
    doc_count = 0

    # dict of topic --> doc count
    docs_per_topic = {}

    # dict of topic --> term count
    terms_per_topic = {}

    # dict of term --> dict of topic --> count
    term_topic_counts = {}

    for doc_id, terms in cd.items():
        doc_count += 1
        terms = terms.split(" ")
        topics = doc_to_topics[doc_id]
        for topic in topics:
            docs_per_topic[topic] = 1 + docs_per_topic.get(topic, 0)

            for term in terms:
                vocab.add(term)
                terms_per_topic[topic] = 1 + terms_per_topic.get(topic, 0)
                if term not in term_topic_counts:
                    term_topic_counts[term] = {}
                term_topic_counts[term][topic] = 1 + \
                    term_topic_counts[term].get(topic, 0)

    prior = {}
    for topic, count in docs_per_topic.items():
        prior[topic] = math.log((count/doc_count))

    condprob = {}
    # count of terms in rel doc that arn't the term

    vocab_len = len(vocab)
    for term in vocab:
        # if not initialized, do that
        if term not in condprob:
            condprob[term] = {}
        # for each topic calculate the cond prob for the given term
        for topic in docs_per_topic.keys():
            # nom = condprob[term].get(topic, 0)+1
            nom = term_topic_counts[term].get(topic, 0) + 1
            denom = terms_per_topic[topic] + vocab_len
            condprob[term][topic] = math.log(nom/denom)

    path = "stem_nostop/NB_stem_nostop.json"
    with open(path, 'wb') as f:
        f.write(orjson.dumps({"prior": prior, "condprob": condprob,
                "doc_to_topics": doc_to_topics, "topics": tp}))

    knn = KNN()
    knn.fit()


# helpermaker()
