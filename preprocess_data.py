"""Preprocesses embedding data for Embedding Comparator.

Computes the local neighborhoods of each object in the embedding model and PCA
dimensionality reduction of all objects. Writes output as JSON.

The embeddings file should contain the embedding vectors, one embedding per line
and each dimension of embedding tab-separated.
The metadata file should contain the label of each embedding, one per line,
in the same order as embeddings_file.

Note: this script should be used to preprocess each model independently.

Example usage:
python preprocess_data.py --base_embeddings_file='model1.bin' \
    --embeddings_file='model2.bin' \
    --base_outfile='data/new/model1.json' \
    --outfile='data/new/model2.json' \
    --max_k=250
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import numpy as np
from sklearn.decomposition import PCA
import sklearn.neighbors as neighbors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from umap import UMAP

from absl import app
from absl import flags
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors

import nltk
from nltk.corpus import stopwords

import re
from tqdm import tqdm
import ipdb

# Round all floats in JSON dump to 5 decimal places.
json.encoder.FLOAT_REPR = lambda x: format(x, '.5f')

METRICS = ['cosine', 'euclidean']
# METRICS = ['cosine']

FLAGS = flags.FLAGS

flags.DEFINE_string('method', 'pca', "Method for dimensionality reduction ('pca','tsne','umap').")
flags.DEFINE_integer(
    'max_k',
    250,
    'Max value of K for defining local neighborhoods (default = 250)')
flags.DEFINE_string('base_embeddings_file', None, 'Path to base embeddings file (bin).')
flags.DEFINE_string('embeddings_file', None, 'Path to embeddings file (bin).')
# flags.DEFINE_string('metadata_file', None, 'Path to metadata file (tsv).')
flags.DEFINE_string('base_outfile', None, 'Path to write base preprocessed data (json).')
flags.DEFINE_string('outfile', None, 'Path to write preprocessed data (json).')


def load_words_embeddings(filepath, base_file):
    # cap_path = datapath(filepath)
    model = load_facebook_vectors(filepath)

    stop_words = stopwords.words('english')

    words = []
    embeddings = []

    if base_file:
        vocab = model.vocab
        f = open('words.txt', 'w')

        print("Initial vocab length: "+str(len(vocab)))
        new_vocab = []
        for word in vocab:
            word = re.sub(r'[^a-z-_]+', '', word.lower())
            word = word.strip()
            if word not in new_vocab and word not in stop_words and word!="" and len(word)>2:
                new_vocab.append(word)

        vocab = new_vocab
        print("Processed vocab length: "+str(len(vocab)))
        for word in vocab:
            words.append(word)
            f.write(word+"\n")
            embeddings.append(model[word].tolist())
        f.close()

    else:
        f = open('words.txt', 'r')
        vocab = f.readlines()
        f.close()
        vocab = [word.strip("\n") for word in vocab]
        for word in vocab:
            words.append(word)
            embeddings.append(model[word].tolist())
    return words, np.array(embeddings)


def load_words(filepath):
    words = []
    with open(filepath, 'r') as f:
        for row in f:
            words.append(row.strip())
    return words


def compute_nearest_neighbors(embeddings, max_k, metric):
    neigh = neighbors.NearestNeighbors(n_neighbors=max_k, metric=metric)
    neigh.fit(embeddings)
    dist, ind = neigh.kneighbors(return_distance=True)
    return ind, dist


def create_nearest_neighbors_dicts(embeddings, max_k, metrics):
    to_return = [
        {metric: None for metric in metrics} for _ in range(len(embeddings))
    ]
    for metric in metrics:
        inds, dists = compute_nearest_neighbors(embeddings, max_k, metric)
        for i, (ind, dist) in enumerate(zip(inds, dists)):
            to_return[i][metric] = {
                'knn_ind': ind.tolist(),
                'knn_dist': dist.tolist(),
            }
    return to_return


def create_preprocessed_data(embeddings, words, nn_dicts, embeddings_pca):
    to_return = []
    for i, (embedding, word, nn_dict, embedding_pca) in enumerate(
        zip(embeddings, words, nn_dicts, embeddings_pca)):
        to_return.append({
            'idx': i,
            'word': word,
            'embedding': list(embedding),
            'nearest_neighbors': nn_dict,
            'embedding_pca': list(embedding_pca),
        })
    return to_return


def run_reduction(embeddings, method):
    if method=="pca":
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(embeddings)
    elif method=="tsne":
        reducer = TSNE(n_components=2)
        reduced = reducer.fit_transform(embeddings)
        reduced = reduced.astype(float)
    elif method=="umap":
        reducer = UMAP(n_components=2)
        reduced = reducer.fit_transform(embeddings)
        reduced = reduced.astype(float)
    return reduced


def write_outfile(outfile_path, preprocessed_data):
    with open(outfile_path, 'w') as f:
        json.dump(preprocessed_data, f, separators=(',', ':'))


def main(argv):

    logging.basicConfig(level=logging.INFO)

    method = FLAGS.method
    base_embeddings_file = FLAGS.base_embeddings_file
    base_outfile_path = FLAGS.base_outfile
    max_k = FLAGS.max_k

    # Load embeddings and words from file.
    words, embeddings = load_words_embeddings(base_embeddings_file, True)
    # words = load_words(metadata_file)

    # Compute nearest neighbors.
    print("Creating nearest neighbors for base...")
    nn_dicts = create_nearest_neighbors_dicts(embeddings, max_k, METRICS)
    print("Running "+method+" for base...")
    embeddings_reduced = run_reduction(embeddings, method)

    print("Preprocess data for base...")
    preprocessed_data = create_preprocessed_data(embeddings, words, nn_dicts, embeddings_reduced)

    # Write preprocessed data to outfile.
    logging.info('Writing data to base outfile: %s' % base_outfile_path)
    write_outfile(base_outfile_path, preprocessed_data)


    outfile_path = FLAGS.outfile
    embeddings_file = FLAGS.embeddings_file

    # Load embeddings and words from file.
    _, embeddings = load_words_embeddings(embeddings_file, False)
    # words = load_words(metadata_file)

    # Compute nearest neighbors.
    print("Creating nearest neighbors...")
    nn_dicts = create_nearest_neighbors_dicts(embeddings, max_k, METRICS)
    print("Running "+method+"...")
    embeddings_reduced = run_reduction(embeddings, method)
    print("Preprocess data...")
    preprocessed_data = create_preprocessed_data(embeddings, words, nn_dicts, embeddings_reduced)

    # Write preprocessed data to outfile.
    logging.info('Writing data to outfile: %s' % outfile_path)
    write_outfile(outfile_path, preprocessed_data)


if __name__ == '__main__':
    flags.mark_flag_as_required('base_embeddings_file')
    flags.mark_flag_as_required('embeddings_file')
    # flags.mark_flag_as_required('metadata_file')
    flags.mark_flag_as_required('base_outfile')
    flags.mark_flag_as_required('outfile')
    app.run(main)
