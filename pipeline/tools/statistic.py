import pandas as pd


class Statistic:
    MAX_LENGTH = 200

    def analyse(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset['tokens'] = dataset.apply(lambda row: self.__count_tokens(row['doc_text']), axis=1)
        dataset['truncated'] = dataset.apply(lambda row: self.__should_truncate(row['tokens']), axis=1)
        return dataset

    @staticmethod
    def calculate(dataset: pd.DataFrame):
        average_tokens_size = dataset['tokens'].mean()
        truncates_docs = dataset['truncated'].sum()
        return average_tokens_size, truncates_docs

    @staticmethod
    def __count_tokens(text):
        return len(text.lower().split())

    @staticmethod
    def __should_truncate(number):
        return 1 if number > Statistic.MAX_LENGTH else 0

