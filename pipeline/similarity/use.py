import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from sklearn.metrics.pairwise import paired_cosine_distances

from .base import Similarity


def split_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class UniversalSentenceEncoderSimilarity(Similarity):
    BATCH_SIZE = 5

    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')

    def score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        sentences1 = dataset['ementa1'].tolist()
        sentences2 = dataset['ementa2'].tolist()
        sentences = list(set(sentences1 + sentences2))

        embeddings = self.__embeddings(sentences)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in sentences1]
        embeddings2 = [emb_dict[sent] for sent in sentences2]

        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        dataset['score'] = cosine_scores
        return dataset

    def __embeddings(self, documents):
        chunks = split_chunks(documents, self.BATCH_SIZE)
        embeddings = None
        for chunk in chunks:
            documents_embed = self.model(chunk)
            documents_embed = documents_embed.numpy()

            if embeddings is None:
                embeddings = documents_embed
            else:
                embeddings = np.concatenate((embeddings, documents_embed), axis=0)
        return embeddings
