import os
import json
import sys

import numpy as np
from pathlib import Path

import tensorflow_text as text
import tensorflow as tf
import tensorflow_hub as hub

from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, TFT5EncoderModel
import settings


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


class Embedder:
    ID = None
    def __init__(self):
        self.name = self.__class__.__name__
        self.version = 1.0
        print("Loaded ", self.ID)

    def embed(self, sentences: [str]):
        raise NotImplementedError


class T5SmallEmbedder(Embedder):
    ID = "t5-base-hugg"
    def __init__(self):
        super(T5SmallEmbedder, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = TFT5EncoderModel.from_pretrained("t5-base")
        self.name = "t5-base-hugg"

    def embed(self, sentences):
        output = np.zeros(shape=(len(sentences), 768), dtype=np.float32)
        tokenizer_output = self.tokenizer(sentences,
                                          truncation=True,
                                          padding="longest",
                                          return_tensors="tf",
                                          return_length=True)

        input_ids = tokenizer_output.input_ids
        seq_lengths = tokenizer_output.length.numpy()

        outputs = self.model(input_ids=input_ids)
        state = outputs.last_hidden_state.numpy()

        for index, elem in enumerate(state):
            seq_length = seq_lengths[index]
            embeddings = state[index][:seq_length]
            avg_embedding = np.average(embeddings, axis=0)
            output[index] = avg_embedding

        return output


class MiniEmbedder(Embedder):
    ID = "small-hugg"
    def __init__(self):
        super(MiniEmbedder, self).__init__()
        self.encoder = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')


    def embed(self, sentences):
        return self.encoder.encode(sentences, batch_size=64, show_progress_bar=False, convert_to_numpy=True)


class T5Embedder(Embedder):
    ID = "t5-base-tfhub"
    def __init__(self):
        super(T5Embedder, self).__init__()
        hub_url = "https://tfhub.dev/google/sentence-t5/st5-base/1"
        encoder = hub.KerasLayer(hub_url)
        self.embedder = encoder

    def embed(self, sentences):
        sentences = tf.constant(sentences)
        return self.embedder(sentences)[0].numpy()

class BertEmbedder:
    ID = "bert-tfhub"
    def __init__(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessor = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        encoder_inputs = preprocessor(text_input)
        encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
            trainable=False)
        outputs = encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]  # [batch_size, 128].
        self.embedding_model = tf.keras.Model(text_input, pooled_output)


    def embed(self, sentences):
        sentences = tf.constant(sentences)
        return self.embedding_model(sentences).numpy()


class T5FullOutputEmbedder:
    ID = "t5-base-hugg"

    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = TFT5EncoderModel.from_pretrained("t5-base")
        self.name = "t5-base-hugg"

    def embed(self, sentence):
        tokenizer_output = self.tokenizer(sentence, return_tensors="tf")

        input_ids = tokenizer_output.input_ids

        outputs = self.model(input_ids=input_ids)
        state = outputs.last_hidden_state.numpy()
        return state

class CachedEmbedder(Embedder):
    def __init__(self, embedder: Embedder, name: str):
        super(CachedEmbedder, self).__init__()
        self.name = name
        self.basedir = self.get_dir(name)
        self.make_dirs()
        self.embedder = embedder
        self.cache_miss = 0
        self.n_requests = 0

        self.embeddings = self.load_embeddings()
        self.text_lines = self.load_text()
        self.text_lookup = self.init_lookup()

        self.tmp_embeddings_store_fp = {}

    def get_cache_hit_ratio(self):
        return 1 - self.cache_miss / self.n_requests

    def make_dirs(self):
        Path(self.basedir).mkdir(parents=True, exist_ok=True)

    def load_embeddings(self):
        embeddings_path = self.get_embeddings_path()
        embeddings = []
        if Path(embeddings_path).exists():
            embeddings = np.load(str(embeddings_path))
        return embeddings

    def load_text(self):
        text_path = self.get_text_path()
        text = []
        if Path(text_path).exists():
            for line in open(text_path):
                line = json.loads(line.strip())
                text.append(line)

        return text

    def embed(self, text_batch):
        missing_indices = {}
        missing_text = []
        missing_embeddings = []

        for index, text in enumerate(text_batch):
            self.n_requests += 1
            idx = self.get_text_idx(text)
            if idx is None:
                self.cache_miss += 1
                missing_text.append(text)
                missing_indices[index] = len(missing_indices)

        if missing_indices:
            missing_embeddings = self.embedder.embed(missing_text)

            for mt, me in zip(missing_text, missing_embeddings):
                self.add_embedding(mt, me)

        embeddings = []
        for index, text in enumerate(text_batch):
            idx = self.get_text_idx(text)
            if idx is None:
                missing_idx = missing_indices[index]
                embedding = missing_embeddings[missing_idx]
            else:
                embedding = self.embeddings[idx]
            embeddings.append(embedding)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        return embeddings

    def get_text_idx(self, text):
        return self.text_lookup.get(text, None)

    def save(self):
        self.verify_existing()
        np.save(self.get_embeddings_path(), self.embeddings)
        with open(self.get_text_path(), 'w') as txtfp:
            for line in self.text_lines:
                x = json.dumps(line) + "\n"
                txtfp.write(x)

        return

    def init_lookup(self):
        self.verify_existing()
        lookup = {}
        for idx, text in enumerate(self.text_lines):
            lookup[text] = idx

        return lookup

    def get_text_path(self):
        return f"{self.basedir}/text.json"

    def get_embeddings_path(self):
        return f"{self.basedir}/embeddings.npy"

    def get_tmp_embeddings_store_path(self):
        return f"{self.basedir}/tmp_embeddings_store.json"

    def get_dir(self, name: str):
        return f"{settings.PathConfig.embeddings_path}/{name}/"

    def add_embedding(self, sentence, embedding):
        self.tmp_embeddings_store_fp[sentence] = embedding

    def verify_existing(self):
        assert len(self.embeddings) == len(self.text_lines)

    def get_pending_commit_size(self):
        return len(self.tmp_embeddings_store_fp)

    def commit(self):
        self.verify_existing()
        # self.tmp_embeddings_store_fp.close()

        embeddings = []

        text_dict = set(self.text_lines)
        for text, embedding in self.tmp_embeddings_store_fp.items():
            if text not in text_dict:
                self.text_lines.append(text)
                embeddings.append(embedding)

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if len(self.embeddings) > 0 and len(embeddings) > 0:
            self.embeddings = np.vstack((self.embeddings, embeddings))
            self.save()

        elif len(embeddings) > 0:
            self.embeddings = embeddings
            self.save()

        self.init_lookup()

class EmbedderFactory:
    ALL = [T5SmallEmbedder, T5Embedder, MiniEmbedder, BertEmbedder]

    @staticmethod
    def list_all():
        return [e.ID for e in EmbedderFactory.ALL]

    @staticmethod
    def check_embedder(name):
        for elem in EmbedderFactory.ALL:
            if elem.ID == name:
                return True
        return False

    @staticmethod
    def get_embedder(name="small-hugg"):
        embedder = None
        for elem in EmbedderFactory.ALL:
            if elem.ID == name:
                embedder = elem
        return embedder

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="embedder_test.txt", help="pass a text file with many lines", required=False)
    parser.add_argument("--embedder", help="pass one of embedders", required=True)

    args = parser.parse_args()

    embedder = EmbedderFactory.get_embedder(args.embedder)
    if embedder is None:
        print(f"Embedder has to be one of {EmbedderFactory.list_all()}")
        sys.exit(0)

    embedder = CachedEmbedder(embedder=embedder(), name="test/")

    sentences = [l.strip() for l in open(args.input)]

    from itertools import groupby
    from tqdm import tqdm

    print("Running", embedder.__class__.__name__)
    embeds = None
    for key, sentence_batch in tqdm(groupby(enumerate(sentences), lambda x: x[0]//64)):
        batch = [l[1] for l in sentence_batch]
        embeds = embedder.embed(batch)
    print(embeds.shape)
    embedder.commit()
