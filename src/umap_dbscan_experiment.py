from embedders import CachedEmbedder
import umap
import umap.plot
import hdbscan
from sklearn.preprocessing import normalize

from sklearn.datasets import load_digits
import pandas as pd

import matplotlib.pyplot as plt


def load_data(name):
    embedder = CachedEmbedder(name=name, embedder=None)
    embeddings = embedder.embeddings
    text = embedder.text_lines
    text_index = {}
    for idx, s in enumerate(text):
        text_index[s] = idx

    return text_index, embeddings

def create_clusters(datadir, cluster_out):
    text_index, embeddings = load_data(datadir)
    text = sorted(text_index, key=text_index.get)

    embeddings = normalize(embeddings)

    umap_embeddings = umap.UMAP(metric='cosine', n_components=2, verbose=True).fit_transform(embeddings)
    cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                              metric='euclidean',
                              cluster_selection_method='leaf').fit(umap_embeddings)

    clustered = sorted(enumerate(text), key=lambda x: cluster.labels_[x[0]])

    # display_clusters(umap_embeddings, cluster)
    fp = open(cluster_out, 'w')
    for i, t in clustered:
        s = f"{cluster.labels_[i]}\t{t}\t{cluster.probabilities_[i]}"
        fp.write(s + "\n")
    fp.close()

def display_clusters(umap_embeddings, cluster):
    result = pd.DataFrame(umap_embeddings, columns=['x', 'y'])
    result['labels'] = cluster.labels_

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="pass a directory", required=True)
    parser.add_argument("--cluster_out", default=None, help="pass a file to put clusters output", required=True)

    args = parser.parse_args()
    create_clusters(args.data, args.cluster_out)