import sklearn.preprocessing
from transformers import T5Tokenizer, TFT5EncoderModel
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


class T5FullOutputEmbedder:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = TFT5EncoderModel.from_pretrained("t5-base")
        self.name = "t5-base-hugg"

    def tokens(self, sentence):
        tokenizer_output = self.tokenizer(sentence, return_tensors="tf")
        input_ids = tokenizer_output.input_ids
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return tokens

    def embed(self, sentence):
        tokenizer_output = self.tokenizer(sentence, return_tensors="tf")
        input_ids = tokenizer_output.input_ids
        outputs = self.model(input_ids=input_ids, output_attentions=True)
        state = outputs.last_hidden_state.numpy()[0]
        attention = outputs.attentions
        return attention

embedder = T5FullOutputEmbedder()

sentences = [input("Enter sentence > ")]

attentions = [embedder.embed(sentence) for sentence in sentences]
tokens = [embedder.tokens(sentence) for sentence in sentences]

token_scores = {}

from prettytable import PrettyTable

for index, sentence in enumerate(sentences):
    table = PrettyTable()
    attention = attentions[index]
    attention = np.sum(np.sum(attention[0], axis=0), axis=0)
    table.field_names = enumerate(tokens[index])
    table.add_rows(attention)
    table.add_column(fieldname="x", column=tokens[index])
    print(table)
