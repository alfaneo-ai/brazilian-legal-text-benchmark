import pandas as pd
import re


class Preprocessor:
    def clear(self, dataset: pd.DataFrame):
        dataset['ementa1'] = dataset.apply(lambda row: self.__clear_text(row['ementa1']), axis=1)
        dataset['ementa2'] = dataset.apply(lambda row: self.__clear_text(row['ementa2']), axis=1)
        return dataset

    def __clear_text(self, text: str) -> str:
        # text = self.__remove_word_ementa(text)
        # text = self.__extract_verbetes(text)
        text = self.__remove_break_lines(text)
        text = self.__to_lowercase(text)
        return text

    @staticmethod
    def __to_lowercase(text: str):
        return text.lower()

    @staticmethod
    def __remove_word_ementa(text: str):
        return re.sub(r'[Ee][Mm][Ee][Nn][Tt][Aa]:?', '', text).strip()

    @staticmethod
    def __remove_break_lines(text: str):
        return re.sub(r'\n', ' ', text).strip()

    @staticmethod
    def __extract_verbetes(text: str):
        return text.split(sep='\n\n', maxsplit=2)[0]
