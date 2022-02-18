import pandas as pd


class Preprocessor:
    def clear(self, dataset: pd.DataFrame):
        dataset['query_text'] = dataset.apply(lambda row: self.__clear_text(row['doc_text']), axis=1)
        dataset['doc_text'] = dataset.apply(lambda row: self.__clear_text(row['doc_text']), axis=1)
        return dataset

    def __clear_text(self, text: str) -> str:
        text = self.__extract_verbetes(text)
        return text

    @staticmethod
    def __extract_verbetes(text: str):
        return text.split(sep='\n', maxsplit=2)[0]
