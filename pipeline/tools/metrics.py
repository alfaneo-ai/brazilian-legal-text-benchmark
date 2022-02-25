import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics:
    def __init__(self):
        self.results = pd.DataFrame({'threshold': [], 'f1': [], 'precision': [], 'recall': []})

    def analyse(self, dataset: pd.DataFrame, threshold):
        y_true = dataset['similarity'].tolist()
        y_pred = dataset['predicted'].tolist()
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        row = {
            'threshold': [threshold],
            'f1': [f1],
            'precision': [precision],
            'recall': [recall]
        }
        case = pd.DataFrame(row)
        self.results = self.results.append(case, ignore_index=True)

    def best_case(self):
        return self.results.iloc[self.results['f1'].argmax()]
