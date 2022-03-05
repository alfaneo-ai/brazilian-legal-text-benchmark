from .preprocess import *
from .tools import *
from .utils import *


class DataAnalyserCommand:

    def __init__(self):
        self.logger = AppLogger()
        self.path_utils = PathUtil()
        self.dataset_utils = DatasetUtils()
        self.preprocessor = Preprocessor()
        self.statistic = Statistic()

    def execute(self):
        self.logger.show('>>>>>>>>>>>> INICIO')

        self.logger.show('')
        self.logger.show('TRAIN DATASET')
        statistics = self.__measure_dataset('train.csv')
        self.__show_statistics(statistics)

        self.logger.show('')
        self.logger.show('DEV DATASET')
        statistics = self.__measure_dataset('dev.csv')
        self.__show_statistics(statistics)

        self.logger.show('')
        self.logger.show('TEST DATASET')
        statistics = self.__measure_dataset('test.csv')
        self.__show_statistics(statistics)

        self.logger.show('>>>>>>>>>>>> TÉRMINO')

    def __measure_dataset(self, filename):
        self.logger.step(f'Analysing dataset {filename}')
        filepath = self.path_utils.build_path('resources', filename)
        query_dataset = self.dataset_utils.from_csv(filepath)
        query_dataset = self.preprocessor.clear(query_dataset)
        stats = self.statistic.calculate(query_dataset)
        return stats

    def __show_statistics(self, statistics):
        self.logger.show(f'Up to 64: {statistics["64"][0]}% - {statistics["64"][1]} samples')
        self.logger.show(f'Up to 128: {statistics["128"][0]}% - {statistics["128"][1]} samples')
        self.logger.show(f'Up to 256: {statistics["256"][0]}% - {statistics["256"][1]} samples')
        self.logger.show(f'Up to 384: {statistics["384"][0]}% - {statistics["384"][1]} samples')
        self.logger.show(f'Up to 512: {statistics["512"][0]}% - {statistics["512"][1]} samples')
        self.logger.show(f'Up to 768: {statistics["768"][0]}% - {statistics["768"][1]} samples')
        self.logger.show(f'Up to 1024: {statistics["1024"][0]}% - {statistics["1024"][1]} samples')
        self.logger.show(f'Greate than 1024: {statistics["more"][0]}% - {statistics["more"][1]} samples')


