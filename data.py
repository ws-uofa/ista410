import os
import json
import numpy as np
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
    """将稀疏矩阵转换为 Stan 高效处理的三元组格式"""
    doc_indices, word_indices, counts = find(dtm)
    # Stan 索引从 1 开始
    return {
        "M": dtm.shape[0],
        "V": dtm.shape[1],
        "N_unique": len(counts),
        "d": (doc_indices + 1).tolist(),
        "w": (word_indices + 1).tolist(),
        "count": counts.tolist()
    }

print("1. 加载目标 QIDs...")
with open(QID_FILE, 'r', encoding='utf-8') as f:
    target_qids = set(line.strip() for line in f if line.strip())

print("2. 加载 MS MARCO...")
msmarco_texts = []
datasets = [ir_datasets.load("msmarco-passage/train"), ir_datasets.load("msmarco-passage/trec-dl-2019/judged")]
found_qids = set()
for ds in datasets:
    for q in ds.queries_iter():
        if q.query_id in target_qids and q.query_id not in found_qids:
            msmarco_texts.append(q.text)
            found_qids.add(q.query_id)
        if len(found_qids) == len(target_qids): break
    if len(found_qids) == len(target_qids): break

print("3. 加载 BEIR 子集...")
beir_texts_dict = {}
out_dir = os.path.join(os.getcwd(), "datasets")
os.makedirs(out_dir, exist_ok=True)
for ds_name in BEIR_DATASETS:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
    data_path = util.download_and_unzip(url, out_dir)
    corpus = GenericDataLoader(data_path).load(split="test")[0]
    # 限制样本以防内存溢出
    texts = [f"{doc.get('title','')} {doc.get('text','')}".strip() for i, doc in enumerate(corpus.values()) if i < 200]
    beir_texts_dict[ds_name] = texts

print("4. 严格基于 MS MARCO 构建词汇表...")
vectorizer = CountVectorizer(stop_words=list(stop_words), token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.9, min_df=2, max_features=2000)
msmarco_dtm = vectorizer.fit_transform(msmarco_texts)

print("5. 转换数据并保存为 JSON...")
# 保存 MS MARCO 训练数据
ms_stan_data = format_for_stan(msmarco_dtm)
ms_stan_data.update({"K": 5, "alpha": [0.1]*5, "beta": [0.1]*ms_stan_data["V"]}) # 设定 5 个主题，使用对称 Dirichlet 先验
with open("data_msmarco.json", "w") as f:
    json.dump(ms_stan_data, f)

# 保存 BEIR 测试数据 (强制使用 MS MARCO 词汇表)
beir_stan_data = {}
for db_name, texts in beir_texts_dict.items():
    dtm = vectorizer.transform(texts)
    beir_stan_data[db_name] = format_for_stan(dtm)

with open("data_beir.json", "w") as f:
    json.dump(beir_stan_data, f)

print("✅ 数据构建完毕，已输出为 data_msmarco.json 和 data_beir.json。")