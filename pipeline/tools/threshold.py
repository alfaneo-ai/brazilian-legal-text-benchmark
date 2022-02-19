import numpy as np
import pandas as pd


class ThresholdTester:
    def __init__(self):
        self.all_queries = pd.DataFrame({'relevant': [], 'score': []})
        self.thresholds = np.arange(0.980, 0.999, 0.001)

    def accumulate(self, dataset: pd.DataFrame):
        self.all_queries = self.all_queries.append(dataset[['relevant', 'score']])

