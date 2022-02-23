import pandas as pd


class Statistic:
    MAX_LENGTH = 200

    def analyse(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset['ementa1_tokens'] = dataset.apply(lambda row: self.__count_tokens(row['ementa1']), axis=1)
        dataset['ementa1_truncated'] = dataset.apply(lambda row: self.__should_truncate(row['ementa1_tokens']), axis=1)
        dataset['ementa2_tokens'] = dataset.apply(lambda row: self.__count_tokens(row['ementa1']), axis=1)
        dataset['ementa2_truncated'] = dataset.apply(lambda row: self.__should_truncate(row['ementa2_tokens']), axis=1)
        return dataset

    @staticmethod
    def calculate(dataset: pd.DataFrame):
        average_tokens_size = (dataset['ementa1_tokens'].mean() + dataset['ementa2_tokens'].mean())/2
        truncates_docs = dataset['ementa1_truncated'].sum() + dataset['ementa2_truncated'].sum()
        return average_tokens_size, truncates_docs

    @staticmethod
    def __count_tokens(text):
        return len(text.lower().split())

    @staticmethod
    def __should_truncate(number):
        return 1 if number > Statistic.MAX_LENGTH else 0

