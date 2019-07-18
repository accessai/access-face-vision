from queue import Empty
import traceback

import cv2

from access_face_vision.access_logger import get_logger
from access_face_vision.component import AccessComponent
from access_face_vision.sink import draw_on_frame


class Display(AccessComponent):

    def __init__(self, cmd_args, in_que, log_que, log_level, kill_app, is_sub_proc=False):
        self.is_sub_proc = is_sub_proc
        self.logger = get_logger(log_que, log_level)

        super(Display, self).__init__(display,
                                      cmd_args=cmd_args,
                                      in_que=in_que,
                                      log_que=log_que,
                                      log_level=log_level,
                                      kill_app=kill_app)

    def __call__(self, obj):
        return draw(obj, self.logger)


def draw(obj, logger):

    frame = draw_on_frame(obj, logger)

    cv2.imshow('Display', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        return False

    return True


def display(cmd_args, in_que, log_que, log_level, kill_proc, kill_app):

    logger = get_logger(log_que, log_level)

    try:
        logger.info('Display process starting')
        while kill_app.value == 0 and kill_proc.value == 0:

            try:
                obj = in_que.get(block=True, timeout=1)
            except Empty as emp:
                logger.debug('Input que to display is empty.')
                continue

            if obj.get('done', False):
                break

            if not draw(obj, logger):
                kill_proc.value = 1
                kill_app.value = 1
                break

    except Exception as ex:
        logger.error(traceback.format_exc())
        kill_app.value = 1
    finally:
        logger.info('Exiting from Display process')
