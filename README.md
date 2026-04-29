data.py --generate data in json file  
data_msmarco.json --generated training data (consist of qids from dev_qids_1000.txt and trec-2019-dl)  
data_beir.json --generated data to infer (consist of BEIR_DATASETS = ["trec-covid", "fiqa", "scidocs", "arguana", "webis-touche2020"])  

--- 
TODO:  
lda_train.stan --Stan script for training  
lda_infer.stan --Stan script for infering  
infer.r  
infer.py  
