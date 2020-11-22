# Embedding Comparator

This repository contains an updated version of the embedding comparator code, originally created by [MITVis](http://vis.mit.edu/). This Python 3 implementation offers loading fasttext models via Gensim, preprocessing via TSNE and UMAP. The tool is very useful for identifying differences between embedding models and tracking evolution of words and their neighborhoods.

### Embedding Comparator Demo

![demo](/images/demo.png)

#### Building upon MITVis embedding comparator

This Python 3 implementation offers:
1. loading fasttext models via Gensim
2. preprocessing via TSNE and UMAP
3. loading and processing all models with one command
4. allow models with different vocabularies. The common vocabulary is extracted by the first embedding model (--base_embeddings_file).
5. given a vocabulary file, visualize only the embeddings for these words (in case your model has a large vocabulary size)

#### Use Cases
1. Compare different ontologies or taxonomies as embeddings (i.e. medical terminologies)
2. Compare temporal versions of the same models
3. Add `--vocab_file='file.txt'` if you want embeddings only for specific words.

#### Live Demo

A demo of the Embedding Comparator is available at: <http://vis.mit.edu/embedding-comparator/>

#### Run Locally

You can also run the Embedding Comparator demo locally by cloning this repository and starting a web server, e.g., by running `python -m http.server`, and then opening <http://localhost:8000/index.html>.

The case study demos in the paper (preprocessed data) are included in the `data/` directory of this repository.
Due to file size constraints, more raw data for these demos (including original embeddings and json files) can be downloaded [here](http://vis.mit.edu/embedding-comparator/raw_data/).

We recommend viewing the Embedding Comparator in Google Chrome.


### Adding your own Models

Adding your own models to the Embedding Comparator involves two steps:

1. Preprocess all fasttext models with the [preprocess_data.py](preprocess_data.py).  
`python preprocess_data.py --base_embeddings_file='model1.bin' --embeddings_file='model2.bin' --base_outfile='data/new/model1.json' --outfile='data/new/model2.json' --max_k=250`. More details and examples in the preprocess.py script.
2. Add `--vocab_file='your_file_path.txt'` if you want embeddings for specific words.
3. Modify the `DATASET_TO_MODELS` object at the top of [embedding_comparator_react.js](embedding_comparator_react.js), adding the model details and path to the processed data (see examples for demo models).


### Acknowledgement

The initial tool is described in the paper:

[Embedding Comparator: Visualizing Differences in Global Structure and Local Neighborhoods via Small Multiples](https://arxiv.org/abs/1912.04853)
<br>
Authors: Angie Boggust, Brandon Carter, Arvind Satyanarayan
