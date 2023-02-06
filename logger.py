import logging
import os
from logging.handlers import RotatingFileHandler
from logging.handlers import TimedRotatingFileHandler

BASEDIR = os.path.abspath(os.path.dirname(__file__))
LOGDIR = os.path.join(BASEDIR, "logs")

class Logger(object):
    logger = None

    @classmethod
    def get_logger(cls):
        if cls.logger is None:
            formatter = logging.Formatter('%(asctime)s - [%(filename)s:%(lineno)s] - %(levelname)s %(message)s ')
            logger = logging.getLogger('ecsbe')
            logger.setLevel(logging.DEBUG)

            console = logging.StreamHandler()
            console.setFormatter(formatter)
            logger.addHandler(console)

            if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)

            file_handler = TimedRotatingFileHandler(os.path.join(LOGDIR, 'log.log'), when="h", interval=1, backupCount=48, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            cls.logger = logger

        return cls.logger