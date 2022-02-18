from abc import ABC, abstractmethod

import pandas as pd


class Similarity(ABC):
    @abstractmethod
    def score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass
