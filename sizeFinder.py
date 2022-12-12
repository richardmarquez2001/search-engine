import orjson
from tqdm import tqdm

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
