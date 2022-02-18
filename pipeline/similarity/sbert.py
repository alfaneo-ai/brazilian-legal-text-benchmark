import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .base import Similarity


class SentenceBertSimilarity(Similarity):
    def __init__(self):
        self.model = SentenceTransformer('ricardo-filho/sbertimbau-base-nli-sts')
        self.model.max_seq_length = 200

    def score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset['score'] = dataset.apply(lambda row: self._calc_score(row), axis=1)
        return dataset

    def _calc_score(self, row):
        query = row['query_text']
        doc = row['doc_text']
        embeddings = self.model.encode([query, doc])
        query_matrix = np.array([embeddings[0]])
        doc_matrix = np.array([embeddings[1]])
        diff = cosine_similarity(query_matrix, doc_matrix)
        return diff.item()


if __name__ == '__main__':
    dataframe = pd.read_csv('teste.csv', sep='|', encoding='utf-8-sig')
    similarity = SentenceBertSimilarity()
    dataframe = similarity.score(dataframe)
    print(dataframe.head(100))
