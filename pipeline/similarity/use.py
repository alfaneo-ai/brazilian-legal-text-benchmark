import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

from .base import Similarity


class UniversalSentenceEncoderSimilarity(Similarity):
    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')

    def score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        documents = dataset['doc_text'].tolist()
        query = dataset['query_text'].iloc[0]
        size = len(documents)

        query_embed = self.model(query)
        query_embed = query_embed.numpy().tolist()[0]
        queries_embed = [query_embed] * size
        queries_embed = np.array(queries_embed)

        documents_embed = self.model(documents)
        documents_embed = documents_embed.numpy()
        diff = cosine_similarity(queries_embed, documents_embed)

        dataset['score'] = diff[0]
        return dataset
