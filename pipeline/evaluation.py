from .preprocess import *
from .similarity import create_similarity
from .tools import *
from .utils import *


class EvaluationCommand:

    def __init__(self, model):
        self.logger = AppLogger()
        self.path_utils = PathUtil()
        self.dataset_utils = DatasetUtils()
        self.statistic = Statistic()
        self.preprocessor = Preprocessor()
        self.threshold = ThresholdTester()
        self.similarity = create_similarity(model)

    def execute(self):
        self.logger.show('>>>>>>>>>>>> INICIO')
        self.logger.start(steps=1)
        query_dataset = self.__measure_similarity()
        self.__show_results(query_dataset)
        self.__show_statistics(query_dataset)
        self.logger.finish()
        self.logger.show('>>>>>>>>>>>> TÉRMINO')

    def __measure_similarity(self):
        self.logger.step(f'Processing')
        filepath = self.path_utils.build_path('resources', 'test.csv')
        query_dataset = self.dataset_utils.from_csv(filepath)
        query_dataset = self.preprocessor.clear(query_dataset)
        query_dataset = self.statistic.analyse(query_dataset)
        query_dataset = self.similarity.score(query_dataset)
        return query_dataset

    def __show_results(self, query_dataset):
        best_case = self.threshold.test(query_dataset)
        self.logger.show('')
        self.logger.show(f'F1: {best_case["f1"]}')
        self.logger.show(f'Precision: {best_case["precision"]}')
        self.logger.show(f'Recall: {best_case["recall"]}')
        self.logger.show(f'Threshold: {best_case["case"]}')

    def __show_statistics(self, query_dataset):
        average_tokens, truncated = self.statistic.calculate(query_dataset)
        self.logger.show('>>>>>>>>>>>>>>>>>>>')
        self.logger.show(f'Average tokens: {average_tokens}')
        self.logger.show(f'Truncated docs: {truncated}')
        self.logger.show(f'Total similarity compares: {len(query_dataset)}')
