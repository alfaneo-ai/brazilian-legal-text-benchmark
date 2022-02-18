import re
import string

import nltk
import pandas as pd
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25L
from .base import Similarity


class Bm25Similarity(Similarity):
    # def __init__(self):
        # self.text_handler = TextHandler()
        # self.index = None

    def score(self, dataset: pd.DataFrame):
        pass
    #     self.__create_index(dataset)
    #     self.__add_score(dataset)
    #
    # def __create_index(self, dataset: pd.DataFrame):
    #     text_corpus = dataset['ementa1'].text.tolist()
    #     tokenized_corpus = self.text_handler.preprocess_list(text_corpus)
    #     self.index = BM25L(tokenized_corpus)
    #
    # def __add_score(self, dataset: pd.DataFrame):
    #     tokens = self.text_handler.preprocess(query_text)
    #     return self.bm25.get_scores(tokens)

# class Bm25Index:
#
#

# class TextHandler:
#     def __init__(self):
#         self.tokenizer = Tokenizer()
#
#     def preprocess_list(self, texts: list) -> list:
#         tokenized_corpus = []
#         for text in texts:
#             tokens = self.preprocess_single(text)
#             tokenized_corpus.append(tokens)
#         return tokenized_corpus
#
#     def preprocess_single(self, text: str) -> list:
#         text = self.tokenizer.spelling(text)
#         tokens = self.tokenizer.split(text)
#         tokens = self.tokenizer.remove_stopwords(tokens)
#         tokens = self.tokenizer.remove_ponctuation(tokens)
#         tokens = self.tokenizer.remove_numbers(tokens)
#         tokens = self.tokenizer.stem(tokens)
#         return tokens
#
#
# class Tokenizer:
#     def __init__(self):
#         nltk.download('stopwords')
#         self.stop_words = nltk.corpus.stopwords.words('portuguese')
#         self.stop_words.append('–')
#         self.stop_words.append('art')
#         self.stop_words.append('ementa')
#         self.stemmer = RSLPStemmer()
#
#     @staticmethod
#     def spelling(text):
#         return re.sub(r'E\sM\sE\sN\sT\sA\s', 'ementa ', text).strip()
#
#     @staticmethod
#     def split(text):
#         return word_tokenize(text, language='portuguese')
#
#     @staticmethod
#     def join(tokens):
#         return ' '.join(tokens)
#
#     def remove_stopwords(self, tokens):
#         return [token for token in tokens if token.lower() not in self.stop_words]
#
#     @staticmethod
#     def remove_ponctuation(tokens):
#         return [token for token in tokens if len(token.strip(string.punctuation)) > 0]
#
#     @staticmethod
#     def remove_numbers(tokens):
#         return [token for token in tokens if re.search(r'\d+\b', token) is None]
#
#     def stem(self, tokens):
#         return [self.stemmer.stem(token) for token in tokens]
