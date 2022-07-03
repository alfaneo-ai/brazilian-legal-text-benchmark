from .utils import *
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import os
from pathlib import Path
import shutil


class EvaluationCommand:

    def __init__(self):
        self.progress = WorkProgress()
        self.path_utils = PathUtil()
        self.datasets_names = ['STJ', 'PJERJ',  'TJMS']
        self.models_names = {
            # 'juridics/bert-base-multilingual-sts-binary',
            'juridics/bertimbau-base-portuguese-sts-binary': False,
            'juridics/bertlaw-base-portuguese-sts-binary': True,

            # 'juridics/bert-base-multilingual-sts-scale',
            # 'juridics/bertimbau-base-portuguese-sts-scale',
            # 'juridics/bertlaw-base-portuguese-sts-scale',

            # 'juridics/bert-base-multilingual-sts-triplet',
            # 'juridics/bertimbau-base-portuguese-sts-triplet',
            # 'juridics/bertlaw-base-portuguese-sts-triplet'
        }

    def execute(self):
        self.progress.show('INICIANDO BENCHMARK')
        if os.path.exists('output'):
            shutil.rmtree('output')
        for dataset_name in self.datasets_names:
            results_path = os.path.join('output', dataset_name)
            Path(results_path).mkdir(parents=True, exist_ok=True)
            for model_name in self.models_names:
                samples = self._prepare_samples(dataset_name, to_lowercase=self.models_names[model_name])

                self.progress.show(100 * '-')
                self.progress.show(f'DATASET {dataset_name} - {len(samples)} rows')
                self.progress.show(100 * '-')

                model = SentenceTransformer(model_name)
                evaluator = BinaryClassificationEvaluator.from_input_examples(samples, show_progress_bar=True,
                                                                              batch_size=8, name='benchmark')
                evaluator(model, output_path=results_path)
        self.progress.show('FINALIZANDO BENCHMARK')

    @staticmethod
    def _prepare_samples(name, to_lowercase=False) -> list:
        filepath = f'resources/{name}.csv'
        dataset = pd.read_csv(filepath, sep='|', encoding='utf-8-sig')
        result = []
        for index, row in dataset.iterrows():
            ementa1 = row['ementa1'].lower() if to_lowercase else row['ementa1']
            ementa2 = row['ementa2'].lower() if to_lowercase else row['ementa2']
            result.append(InputExample(texts=[ementa1, ementa2], label=int(row['similarity'])))
            result.append(InputExample(texts=[ementa2, ementa1], label=int(row['similarity'])))
        return result
