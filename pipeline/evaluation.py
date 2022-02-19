from .preprocess import *
from .similarity import create_similarity
from .tools import *
from .utils import *


class EvaluationCommand:
    ROOT_PATH = '/home/cviegas/Workspace/mestrado/brazilian-legal-text-dataset/output/sample'

    def __init__(self, model):
        self.logger = AppLogger()
        self.path_utils = PathUtil()
        self.dataset_utils = DatasetUtils()
        self.statistic = Statistic()
        self.preprocessor = Preprocessor()
        self.merger = Merger()
        self.threshold = ThresholdTester()
        self.similarity = create_similarity(model)

    def execute(self):
        self.logger.show('>>>>>>>>>>>> INICIO')
        filepaths = self.__find_query_files()
        for filepath in filepaths:
            self.__measure_similarity(filepath)
        self.__show_results()
        self.logger.show('>>>>>>>>>>>> TÉRMINO')

    def __find_query_files(self):
        filepaths = self.path_utils.get_files(self.ROOT_PATH, '*.csv')
        self.logger.start(len(filepaths))
        return filepaths

    def __measure_similarity(self, filepath):
        self.logger.step(f'Processing {filepath}')
        query_dataset = self.dataset_utils.from_csv(filepath)
        query_dataset = self.preprocessor.clear(query_dataset)
        query_dataset = self.statistic.analyse(query_dataset)
        query_dataset = self.similarity.score(query_dataset)
        self.merger.merge(query_dataset)

    def __show_results(self):
        average_tokens, truncated = self.statistic.calculate(self.merger.dataset)
        self.logger.show(f'Average tokens: {average_tokens}')
        self.logger.show(f'Truncated docs: {truncated}')
        self.logger.show(f'Total similarity compares: {len(self.merger.dataset)}')



