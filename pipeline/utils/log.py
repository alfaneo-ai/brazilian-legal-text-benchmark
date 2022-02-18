import logging


def create_logger():
    console_handler = logging.StreamHandler()
    lineformat = '[%(asctime)s] | %(levelname)s | [%(process)d - %(processName)s]: %(message)s'
    dateformat = '%d-%m-%Y %H:%M:%S'
    logging.basicConfig(format=lineformat, datefmt=dateformat, level=20, handlers=[console_handler])
    return logging.getLogger()


class Logger:
    def __init__(self):
        self.logger = create_logger()

    def info(self, message: str):
        self.logger.info(message)
