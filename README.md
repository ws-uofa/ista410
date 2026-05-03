# Scripts for Group 1's Final Project
## Bayesian Topic Modeling and Uncertainty Quantification across MS MARCO and BEIR

### Team Members:
[Wilson Sun](https://wilsun.io), Yiting Yang, Yubo Zhou

### Environment Setup:
```bash
pip install numpy scipy pandas matplotlib scikit-learn nltk ir_datasets beir cmdstanpy
```

### Main Scripts:
* ```data.py``` Downloads, preprocesses, and formats in-domain (MS MARCO) and out-of-domain (BEIR) text corpora into sparse, Stan-compatible JSON structures for efficient Latent Dirichlet Allocation (LDA) modeling.
* ```lda_train.stan```: A Stan script that defines the generative LDA model, utilizing multi-threaded reduce_sum to train and sample both document-topic (theta) and topic-word (phi) distributions.
* ```lda_infer.stan```: A specialized Stan inference model that estimates document-topic distributions for unseen target datasets while holding the pre-trained topic-word distributions fixed.
* ```main.py```: The main orchestration pipeline that compiles the Stan models, coordinates in-domain training and highly concurrent out-of-domain MCMC inference, and generates model diagnostic and uncertainty visualizations.
