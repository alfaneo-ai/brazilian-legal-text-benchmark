from .bm25 import Bm25Similarity
from .sbert import SentenceBertSimilarity


def create_similarity(model):
    if model == 'sbert':
        return SentenceBertSimilarity()
    elif model == 'bm25':
        return Bm25Similarity()
