import os
import shutil
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from sklearn.utils import shuffle

from .utils import *


class EvaluationCommand:

    def __init__(self):
        self.progress = WorkProgress()
        self.path_utils = PathUtil()
        self.datasets_names = ['PJERJ', 'TJMS']
        self.models_names = {
            'bert-base-multilingual-cased': {'lower': False, 'group': 1, 'id': 1},
            'neuralmind/bert-base-portuguese-cased': {'lower': False, 'group': 1, 'id': 2},
            'juridics/bertimbaulaw-base-portuguese-cased': {'lower': False, 'group': 1, 'id': 3},
            'juridics/bertlaw-base-portuguese-uncased': {'lower': True, 'group': 1, 'id': 4},

            'juridics/bert-base-multilingual-sts-scale': {'lower': False, 'group': 2, 'id': 5},
            'juridics/bertimbau-base-portuguese-sts-scale': {'lower': False, 'group': 2, 'id': 6},
            'juridics/bertimbaulaw-base-portuguese-sts-scale': {'lower': False, 'group': 2, 'id': 7},
            'juridics/bertlaw-base-portuguese-sts-scale': {'lower': True, 'group': 2, 'id': 8}
        }

    def execute(self):
        self.progress.show('INICIANDO BENCHMARK')
        if os.path.exists('output'):
            shutil.rmtree('output')
        for dataset_name in self.datasets_names:
            results_path = os.path.join('output', dataset_name)
            Path(results_path).mkdir(parents=True, exist_ok=True)
            for model_name in self.models_names:
                lower = self.models_names[model_name]['lower']
                group = self.models_names[model_name]['group']
                id = self.models_names[model_name]['id']
                samples = self._prepare_samples(dataset_name, to_lowercase=lower)

                self.progress.show(100 * '-')
                self.progress.show(f'DATASET {dataset_name} - {len(samples)} rows')
                self.progress.show(100 * '-')

                model = self._create_model(model_name, id)
                evaluator = BinaryClassificationEvaluator.from_input_examples(samples, show_progress_bar=True,
                                                                              batch_size=8, name='benchmark')
                evaluator(model, output_path=results_path, epoch=group, steps=id)
        self.progress.show('FINALIZANDO BENCHMARK')

    @staticmethod
    def _create_model(model_name, id):
        if id > 4:
            return SentenceTransformer(model_name)
        word_embedding_model = models.Transformer(model_name, max_seq_length=384)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        return SentenceTransformer(modules=[word_embedding_model, pooling_model])

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

        result = shuffle(result, random_state=0)
        return result
