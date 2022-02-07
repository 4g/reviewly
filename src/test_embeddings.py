from pathlib import Path
import json
import numpy as np
import random
from sklearn.preprocessing import normalize

def load_data(data_dir):
    output_dir = Path(data_dir)
    text_json = output_dir / "text.json"
    embeddings_npy = output_dir / "embeddings.npy"
    text = []
    for line in open(text_json):
        line = json.loads(line.strip())
        text.append(line)

    embeddings = np.load(str(embeddings_npy))

    n_samples = len(text)
    print(f"Loaded embeddings: {embeddings.shape}, and text {n_samples}")

    embeddings = normalize(embeddings)
    text_index = {}
    for index, line in enumerate(text):
        text_index[line] = index

    return text_index, embeddings


def test_clusters(datadir):
    text_index, embeddings = load_data(datadir)
    text = sorted(text_index, key=text_index.get)

    for i in range(100):
        index = np.random.randint(0, len(text))
        embedding = embeddings[index]
        dots = embeddings @ embedding
        index_sort = np.argsort(dots)[::-1]
        top_k = index_sort[:10]
        print("----------------")
        print(index, text[index])
        for top_index in top_k:
            print(dots[top_index], top_index, text[top_index])



if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="pass a directory", required=True)

    args = parser.parse_args()
    test_clusters(args.data)