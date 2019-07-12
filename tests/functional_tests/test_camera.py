from multiprocessing import Queue, Value
from time import sleep

from access_face_vision.source.camera import Camera
from access_face_vision.utils import create_parser
from access_face_vision import access_logger

LOG_LEVEL = 'debug'
logger, log_que, que_listener = access_logger.set_main_process_logger(LOG_LEVEL)


def test_camera():
    logger.info('Starting Camera test')
    cmd_args = create_parser()
    camera = Camera(cmd_args, Queue(), log_que, LOG_LEVEL, Value('i',0), draw_frames=True)
    camera.start()
    sleep(60)
    camera.stop()
    logger.info('Camera test completed')
    que_listener.stop()


if __name__ == '__main__':
    test_camera()
