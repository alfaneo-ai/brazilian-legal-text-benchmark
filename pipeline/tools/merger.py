import pandas as pd


class Merger:
    def __init__(self):
        self.dataset = pd.DataFrame({'relevant': [], 'score': [], 'tokens': [], 'truncated': []})

    def merge(self, dataset: pd.DataFrame):
        self.dataset = self.dataset.append(dataset[['relevant', 'score', 'tokens', 'truncated']])
