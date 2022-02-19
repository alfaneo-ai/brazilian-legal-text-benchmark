import numpy as np
import pandas as pd


class ThresholdTester:
    def __init__(self):
        self.thresholds = np.arange(0.980, 0.999, 0.001)

    def test(self, dataset: pd.DataFrame):
        pass

