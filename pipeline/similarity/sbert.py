import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances

from .base import Similarity


class SentenceBertSimilarity(Similarity):
    def __init__(self):
        self.model = SentenceTransformer(
            '/home/cviegas/Workspace/mestrado/brazilian-legal-text-bert/output/finetunning-1_epochs')
        self.model.max_seq_length = 256

    def score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        sentences1 = dataset['ementa1'].tolist()
        sentences2 = dataset['ementa2'].tolist()

        sentences = list(set(sentences1 + sentences2))
        embeddings = self.model.encode(sentences,
                                       batch_size=5,
                                       show_progress_bar=True,
                                       convert_to_numpy=True)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in sentences1]
        embeddings2 = [emb_dict[sent] for sent in sentences2]

        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        dataset['score'] = cosine_scores
        return dataset
