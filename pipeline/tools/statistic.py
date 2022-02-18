import pandas as pd


class Statistic:
    MAX_LENGTH = 200

    def __init__(self):
        self.total_files = 0
        self.average_tokens_size = 0
        self.total_truncated_documents = 0

    def analyse(self, dataset: pd.DataFrame):
        dataset['doc_tokens'] = dataset.apply(lambda row: self.__count_tokens(row['doc_text']), axis=1)
        dataset['doc_truncated'] = dataset.apply(lambda row: self.__should_truncate(row['doc_tokens']), axis=1)
        mean_docs = dataset['doc_tokens'].mean()
        truncates_docs = dataset['doc_truncated'].sum()
        self.average_tokens_size = (self.average_tokens_size + mean_docs)/2
        self.total_truncated_documents += truncates_docs
        self.total_files += 1

    @staticmethod
    def __count_tokens(text):
        return len(text.lower().split())

    @staticmethod
    def __should_truncate(number):
        return 1 if number > Statistic.MAX_LENGTH else 0

