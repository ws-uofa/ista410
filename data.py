import os
import json
import numpy
import ir_datasets
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from scipy.sparse import find

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

QID_FILE = "dev_qids_1000.txt"
BEIR_DATASETS = ["trec-covid", "fiqa", "scidocs", "arguana", "webis-touche2020"]

def format_for_stan(dtm):
    doc_indices, word_indices, counts = find(dtm)
    return {
        "M": dtm.shape[0],
        "V": dtm.shape[1],
        "N_unique": len(counts),
        "d": (doc_indices + 1).tolist(),
        "w": (word_indices + 1).tolist(),
        "count": counts.tolist()
    }

with open(QID_FILE, 'r', encoding='utf-8') as f:
    target_qids = set(line.strip() for line in f if line.strip())

qid_to_pos_doc = {}
dataset_names = ["msmarco-passage/train", "msmarco-passage/trec-dl-2019/judged"]

for ds_name in dataset_names:
    ds = ir_datasets.load(ds_name)
    for qrel in ds.qrels_iter():
        if qrel.query_id in target_qids and qrel.relevance > 0:
            if qrel.query_id not in qid_to_pos_doc:
                qid_to_pos_doc[qrel.query_id] = qrel.doc_id

target_doc_ids = set(qid_to_pos_doc.values())
print(f"obtained {len(target_doc_ids)} docs")

msmarco_texts = []
corpus = ir_datasets.load("msmarco-passage")
for doc in corpus.docs_iter():
    if doc.doc_id in target_doc_ids:
        msmarco_texts.append(doc.text)
        target_doc_ids.remove(doc.doc_id)
    if not target_doc_ids:
        break

beir_texts_dict = {}
out_dir = os.path.join(os.getcwd(), "datasets")
os.makedirs(out_dir, exist_ok=True)
for ds_name in BEIR_DATASETS:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
    data_path = util.download_and_unzip(url, out_dir)
    corpus_beir = GenericDataLoader(data_path).load(split="test")[0]
    texts = [f"{doc.get('title','')} {doc.get('text','')}".strip() for i, doc in enumerate(corpus_beir.values()) if i < 200]
    beir_texts_dict[ds_name] = texts

vectorizer = CountVectorizer(stop_words=list(stop_words), token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.9, min_df=2, max_features=2000)
msmarco_dtm = vectorizer.fit_transform(msmarco_texts)

ms_stan_data = format_for_stan(msmarco_dtm)
ms_stan_data.update({"K": 5, "alpha": [1.1]*5, "beta": [1.1]*ms_stan_data["V"]}) 

with open("data_msmarco.json", "w") as f:
    json.dump(ms_stan_data, f)

beir_stan_data = {}
for db_name, texts in beir_texts_dict.items():
    dtm = vectorizer.transform(texts)
    beir_stan_data[db_name] = format_for_stan(dtm)

with open("data_beir.json", "w") as f:
    json.dump(beir_stan_data, f)

print("Data constructed.")