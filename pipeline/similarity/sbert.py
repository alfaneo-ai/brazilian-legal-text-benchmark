import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .base import Similarity


class SentenceBertSimilarity(Similarity):
    def __init__(self):
        self.model = SentenceTransformer('/home/cviegas/Workspace/mestrado/brazilian-legal-text-bert/output/finetunning-2022-02-21_08-30-29')
        self.model.max_seq_length = 256

    def score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        documents = dataset['ementa1'].tolist()
        queries = dataset['ementa2'].tolist()

        queries_embed = self.model.encode(queries)
        queries_embed = np.array(queries_embed)

        documents_embed = self.model.encode(documents)
        documents_embed = np.array(documents_embed)
        diff = cosine_similarity(queries_embed, documents_embed)

        dataset['score'] = diff[0]
        return dataset
