import orjson
import gzip
import regex as re
from tqdm import tqdm

count = 0
urls = {}

datasetPath = r"C:\Users\jkyle\Desktop\CPS842\trec_corpus_5000.jsonl.gz"
with gzip.open(datasetPath, 'rb') as f:
    for line in tqdm(f):
        count += 1
        doc = orjson.loads(line)
        id = str(doc['id'])
        contents = doc['contents']
        url = re.search("about=\"(.*?)\"", contents)
        if url:
            url = url.group()[7:-1]
        else:
            print(id)
            url = ""

        urls[id] = url

        if count == 100000:
            break
    path = "stem_nostop/urls_stem_nostop.json"
    with open(path, 'wb') as f:
        f.write(orjson.dumps(urls))
