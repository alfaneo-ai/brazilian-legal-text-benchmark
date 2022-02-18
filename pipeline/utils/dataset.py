import pandas as pd


class DatasetUtils:
    SEPARATOR = '|'
    ENCODING = 'utf-8-sig'

    def from_csv(self, filepath: str, index_col=None) -> pd.DataFrame:
        if index_col:
            return pd.read_csv(filepath, index_col=index_col, sep=self.SEPARATOR, encoding=self.ENCODING)
        return pd.read_csv(filepath, sep=self.SEPARATOR, encoding=self.ENCODING)

    def to_csv(self, dataset: pd.DataFrame, filepath: str, index_label='index'):
        dataset.to_csv(filepath, sep=self.SEPARATOR, encoding=self.ENCODING, index_label=index_label)
