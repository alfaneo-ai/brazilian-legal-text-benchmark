from .tools import *
from .utils import *


class EvaluationCommand:
    ROOT_PATH = '/home/cviegas/Workspace/mestrado/brazilian-legal-text-dataset/output/query'

    def __init__(self, model):
        self.work_progress = WorkProgress()
        self.path_utils = PathUtil()
        self.dataset_utils = DatasetUtils()
        self.model = model

    def execute(self):
        self.work_progress.show('>>>>>>>>>>>> INICIO')
        filepaths = self.prepare_queries()
        for filepath in filepaths:
            print(filepath)
        self.work_progress.show('>>>>>>>>>>>> TÉRMINO')

    def prepare_queries(self):
        return self.path_utils.get_files(self.ROOT_PATH, '*.csv')
