from .utils import *
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample


class EvaluationCommand:

    def __init__(self):
        self.progress = WorkProgress()
        self.path_utils = PathUtil()

    def execute(self):
        self.progress.show('>>>>>>>>>>>> INICIO')
        samples = self._prepare_samples(just_a_sample=True)
        evaluator = BinaryClassificationEvaluator.from_input_examples(samples, show_progress_bar=True,
                                                                      batch_size=8, name=f'benchmark_results')
        model = SentenceTransformer('juridics/bertlaw-base-portuguese-triplet-sts')
        evaluator(model, output_path='output')
        self.progress.show('>>>>>>>>>>>> TÉRMINO')

    @staticmethod
    def _prepare_samples(just_a_sample=False) -> list:
        filepath = 'resources/full.csv'
        dataset = pd.read_csv(filepath, sep='|', encoding='utf-8-sig')
        if just_a_sample:
            dataset = dataset.sample(frac=0.01)
        result = []
        for index, row in dataset.iterrows():
            result.append(InputExample(texts=[row['ementa1'], row['ementa2']], label=int(row['similarity'])))
            result.append(InputExample(texts=[row['ementa2'], row['ementa1']], label=int(row['similarity'])))
        return result
