# Embedding Comparator

This repository contains an updated version of the embedding comparator code, originally created by [MITVis](http://vis.mit.edu/). This Python 3 implementation offers loading fasttext models via Gensim, preprocessing via TSNE and UMAP.

### Embedding Comparator Demo

![demo](/images/demo.png)

#### Live Demo

A demo of the Embedding Comparator is available at: <http://vis.mit.edu/embedding-comparator/>

#### Run Locally

You can also run the Embedding Comparator demo locally by cloning this repository and starting a web server, e.g., by running `python -m http.server`, and then opening <http://localhost:8000/index.html>.

The case study demos in the paper (preprocessed data) are included in the `data/` directory of this repository.
Due to file size constraints, raw data for these demos (including original embeddings and words in tsv format) can be downloaded [here](http://vis.mit.edu/embedding-comparator/raw_data/).

We recommend viewing the Embedding Comparator in Google Chrome.


### Adding your own Models

Adding your own models to the Embedding Comparator involves two steps:

1. Preprocess all fasttext models with the [preprocess_data.py](preprocess_data.py).  
`python preprocess_data.py --base_embeddings_file='model1.bin' --embeddings_file='model2.bin' --base_outfile='data/new/model1.json' --outfile='data/new/model2.json' --max_k=250`. More details and examples in the preprocess.py script.
2. Modify the `DATASET_TO_MODELS` object at the top of [embedding_comparator_react.js](embedding_comparator_react.js), adding the model details and path to the processed data (see examples for demo models).


### Acknowledgement

The initial tool is described in the paper:

[Embedding Comparator: Visualizing Differences in Global Structure and Local Neighborhoods via Small Multiples](https://arxiv.org/abs/1912.04853)
<br>
Authors: Angie Boggust, Brandon Carter, Arvind Satyanarayan
