from .preprocess import *
from .tools import *
from .utils import *


class EvaluationCommand:
    ROOT_PATH = '/home/cviegas/Workspace/mestrado/brazilian-legal-text-dataset/output/query'

    def __init__(self, model):
        self.logger = AppLogger()
        self.path_utils = PathUtil()
        self.dataset_utils = DatasetUtils()
        self.statistic = Statistic()
        self.preprocessor = Preprocessor()
        self.model = model

    def execute(self):
        self.logger.show('>>>>>>>>>>>> INICIO')
        filepaths = self.__get_file_queries()
        for filepath in filepaths:
            self.__measure_similarity(filepath)
        self.__show_results()
        self.logger.show('>>>>>>>>>>>> TÉRMINO')

    def __get_file_queries(self):
        filepaths = self.path_utils.get_files(self.ROOT_PATH, '*.csv')
        self.logger.start(len(filepaths))
        return filepaths

    def __measure_similarity(self, filepath):
        self.logger.step(f'Processing {filepath}')
        query_dataset = self.dataset_utils.from_csv(filepath)
        query_dataset = self.preprocessor.clear(query_dataset)
        self.statistic.analyse(query_dataset)

    def __show_results(self):
        self.logger.show(f'Total files: {self.statistic.total_files}')
        self.logger.show(f'Average tokens: {self.statistic.average_tokens_size}')
        self.logger.show(f'Truncated docs: {self.statistic.total_truncated_documents}')


