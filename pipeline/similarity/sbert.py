import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .base import Similarity


class SentenceBertSimilarity(Similarity):
    def __init__(self):
        # self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        # self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        # self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        # self.model = SentenceTransformer('ricardo-filho/bert-portuguese-cased-nli-assin-assin-2')
        # self.model = SentenceTransformer('ricardo-filho/sbertimbau-base-nli-sts')
        self.model = SentenceTransformer('ricardo-filho/sbertimbau-large-nli-sts')

        self.model.max_seq_length = 200

    def score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        documents = dataset['doc_text'].tolist()
        query = dataset['query_text'].iloc[0]
        size = len(documents)

        query_embed = self.model.encode(query)
        queries_embed = [query_embed] * size
        queries_embed = np.array(queries_embed)

        documents_embed = self.model.encode(documents)
        documents_embed = np.array(documents_embed)
        diff = cosine_similarity(queries_embed, documents_embed)

        dataset['score'] = diff[0]
        return dataset
