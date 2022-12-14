import csv
import orjson
import math


def helpermaker():
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
