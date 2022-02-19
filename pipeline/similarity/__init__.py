from .bm25 import Bm25Similarity
from .sbert import SentenceBertSimilarity
from .use import UniversalSentenceEncoderSimilarity


def create_similarity(model):
    if model == 'sbert':
        return SentenceBertSimilarity()
    elif model == 'use':
        return UniversalSentenceEncoderSimilarity()
    elif model == 'bm25':
        return Bm25Similarity()
