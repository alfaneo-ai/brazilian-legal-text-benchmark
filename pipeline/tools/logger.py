import logging
import datetime


def create_logger():
    console_handler = logging.StreamHandler()
    lineformat = '[%(asctime)s] | %(levelname)s | [%(process)d - %(processName)s]: %(message)s'
    dateformat = '%d-%m-%Y %H:%M:%S'
    logging.basicConfig(format=lineformat, datefmt=dateformat, level=20, handlers=[console_handler])
    return logging.getLogger()


class AppLogger:
    def __init__(self):
        self.logger = create_logger()
        self.total_steps = 0
        self.current_step = 0
        self.started = None

    def show(self, message):
        self.logger.info(message)

    def start(self, steps):
        self.total_steps = steps
        self.current_step = 0
        self.started = datetime.datetime.now().replace(microsecond=0)

    def step(self, message):
        self.current_step += 1
        self.logger.info(f'{self.current_step} de {self.total_steps} - {message}')

    def finish(self):
        finish = datetime.datetime.now().replace(microsecond=0)
        duration = finish - self.started
        self.logger.info(f'Duration {duration}')