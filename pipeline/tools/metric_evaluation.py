import os
import uuid

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def generate_execution_id():
    return uuid.uuid1()


class MetricEvaluation:
    def __init__(self, output_path):
        self.query_metrics = pd.DataFrame({'assunto': [], 'acordao_id': [], 'f1': [], 'precision': [], 'recall': []})
        self.output_path = output_path

    def add(self, group_name, query_id, y_true, y_pred):
        """Calcula os scores f1, precision e recall para a query informada
            >>> metric.add('bla', '1', [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
            (1.0, 1.0, 1.0)
        """
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        row = {
            'assunto': group_name,
            'acordao_id': [str(query_id)],
            'f1': [f1],
            'precision': [precision],
            'recall': [recall]
        }
        query = pd.DataFrame(row)
        self.query_metrics = self.query_metrics.append(query, ignore_index=True)
        return f1, precision, recall

    def macro_mean(self):
        """Calcula a macro média das métricas
            >>> metric.add('bla1', 'q1', [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
            (1.0, 1.0, 1.0)
            >>> metric.add('bla2', 'q2', [1, 1, 1, 1, 1], [0, 0, 0, 0, 0])
            (0.0, 0.0, 0.0)
            >>> metric.macro_mean()
            (0.5, 0.5, 0.5)
        """
        f1, precision, recall = self.query_metrics[['f1', 'precision', 'recall']].mean()
        return f1, precision, recall

    def save(self, execution_id):
        execution_id = execution_id if execution_id else generate_execution_id()
        executaion_filepath = os.path.join(self.output_path, f'{execution_id}.csv')
        self.query_metrics.to_csv(executaion_filepath, index_label='index')

        summary, summary_filepath = self.__prepare_summary_output()
        summary_row = self.prepare_summary_metrics(execution_id)
        self.save_summary(summary, summary_filepath, summary_row)

    def __prepare_summary_output(self):
        summary_filepath = os.path.join(self.output_path, f'summary.csv')
        if os.path.isfile(summary_filepath):
            summary = pd.read_csv(summary_filepath, index_col='index')
        else:
            summary = pd.DataFrame({'execution': [], 'f1': [], 'precision': [], 'recall': []})
        return summary, summary_filepath

    def prepare_summary_metrics(self, execution_id):
        f1, precision, recall = self.macro_mean()
        row = {
            'execution': execution_id,
            'f1': [f1],
            'precision': [precision],
            'recall': [recall]
        }
        summary_row = pd.DataFrame(row)
        return summary_row

    @staticmethod
    def save_summary(summary, summary_filepath, summary_row):
        summary = summary.append(summary_row, ignore_index=True)
        summary = summary.sort_values(by='f1', ascending=False)
        summary.to_csv(summary_filepath, index_label='index')


if __name__ == "__main__":
    import doctest

    doctest.testmod(extraglobs={'metric': MetricEvaluation()})
