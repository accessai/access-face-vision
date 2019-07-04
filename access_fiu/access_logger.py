import sys
import logging
import logging.handlers

FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s"


def get_listener_logger(que):

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(FORMAT))
    sh.setLevel(logging.INFO)
    que_listener = logging.handlers.QueueListener(que, sh)

    return que_listener


def conf_worker_logger(que):
    que_handler = logging.handlers.QueueHandler(que)
    que_handler.setLevel(logging.INFO)
    que_handler.setFormatter(logging.Formatter(FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    if not root_logger.hasHandlers():
        root_logger.addHandler(que_handler)