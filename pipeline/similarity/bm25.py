import re
import string

import nltk
import numpy as np
import pandas as pd
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25L

from .base import Similarity


class Bm25Similarity(Similarity):
    def __init__(self):
        self.index = None
        self.text_processor = Textprocessor()

    def score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        self.__create_index(dataset)
        dataset = self.__add_score_column(dataset)
        dataset = self.__normalize_score(dataset)
        return dataset

    def __create_index(self, dataset: pd.DataFrame):
        text_corpus = dataset['doc_text'].tolist()
        tokenized_corpus = [self.__clean(text) for text in text_corpus]
        self.index = BM25L(tokenized_corpus)

    def __add_score_column(self, dataset: pd.DataFrame):
        query_text = dataset['query_text'].iloc[0]
        query_tokens = self.__clean(query_text)
        scores = self.index.get_scores(query_tokens)
        scores = np.array(scores)
        dataset['score'] = scores
        return dataset

    def __clean(self, text: str) -> str:
        text = self.text_processor.spelling(text)
        tokens = self.text_processor.tokenize(text)
        tokens = self.text_processor.remove_stopwords(tokens)
        tokens = self.text_processor.remove_ponctuation(tokens)
        tokens = self.text_processor.remove_numbers(tokens)
        tokens = self.text_processor.stem(tokens)
        return tokens

    @staticmethod
    def __normalize_score(dataset: pd.DataFrame):
        dataset['score'] = (dataset['score'] - dataset['score'].min()) / (
                    dataset['score'].max() - dataset['score'].min())
        return dataset


class Textprocessor:
    def __init__(self):
        nltk.download('stopwords')
        self.stop_words = nltk.corpus.stopwords.words('portuguese')
        self.stop_words.append('–')
        self.stop_words.append('art')
        self.stop_words.append('ementa')
        self.stemmer = RSLPStemmer()

    @staticmethod
    def spelling(text):
        return re.sub(r'E\sM\sE\sN\sT\sA\s', 'ementa ', text).strip()

    @staticmethod
    def tokenize(text):
        return word_tokenize(text, language='portuguese')

    @staticmethod
    def join(tokens):
        return ' '.join(tokens)

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token.lower() not in self.stop_words]

    @staticmethod
    def remove_ponctuation(tokens):
        return [token for token in tokens if len(token.strip(string.punctuation)) > 0]

    @staticmethod
    def remove_numbers(tokens):
        return [token for token in tokens if re.search(r'\d+\b', token) is None]

    def stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

