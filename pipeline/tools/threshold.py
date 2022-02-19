import numpy as np
import pandas as pd

from .metrics import Metrics


class ThresholdTester:
    def __init__(self):
        self.thresholds = np.arange(0.1, 0.9, 0.1)
        self.metrics = Metrics()

    def test(self, dataset: pd.DataFrame):
        for threshold in self.thresholds:
            dataset['predicted'] = dataset.apply(lambda row: self.__is_relevant(row['score'], threshold), axis=1)
            self.metrics.analyse(dataset, str(threshold))
        return self.metrics.best_case()

    @staticmethod
    def __is_relevant(score, threshold):
        return 1 if score >= threshold else 0
