import sys
import logging
import logging.handlers

FORMAT = "%(asctime)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s"


def get_listener_logger(que):

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(FORMAT))
    sh.setLevel(logging.INFO)
    que_listener = logging.handlers.QueueListener(que, sh)

    return que_listener


def conf_worker_logger(que):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    if not root_logger.hasHandlers():
        que_handler = logging.handlers.QueueHandler(que)
        que_handler.setLevel(logging.INFO)
        que_handler.setFormatter(logging.Formatter(FORMAT))
        root_logger.addHandler(que_handler)


def get_log_level_code(log):
    if log.lower() == 'info':
        return logging.INFO
    elif log.lower() == 'debug':
        return logging.DEBUG
    elif log.lower() == 'warn':
        return logging.WARNING
    else:
        return logging.ERROR


def set_main_process_logger(log_level):
    from multiprocessing import Queue

    log_que = Queue(-1)
    que_listener = get_listener_logger(log_que)
    conf_worker_logger(log_que)
    logger = logging.getLogger(__name__)
    logger.setLevel(get_log_level_code(log_level))
    que_listener.start()

    return logger, log_que, que_listener


def get_logger(log_que, log_level):
    conf_worker_logger(log_que)
    logger = logging.getLogger(__name__)
    logger.setLevel(get_log_level_code(log_level))

    return logger