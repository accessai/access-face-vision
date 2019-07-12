from multiprocessing import Queue, Value
from time import sleep

from access_face_vision.source.camera import Camera
from access_face_vision.face_detector import FaceDetector
from access_face_vision.sink.display import Display
from access_face_vision.utils import create_parser
from access_face_vision import access_logger

LOG_LEVEL = 'debug'
logger, log_que, que_listener = access_logger.set_main_process_logger(LOG_LEVEL)


def test_face_detector_pipeline():
    logger.info('Starting Face Detector Pipeline test')
    cmd_args = create_parser()
    camera_out_que = Queue()
    face_detector_out_que = Queue()
    kill_app = Value('i', 0)

    camera = Camera(cmd_args, camera_out_que, log_que, LOG_LEVEL, kill_app, draw_frames=False)
    face_detector = FaceDetector(cmd_args, camera_out_que,
                                 face_detector_out_que,
                                 log_que,LOG_LEVEL,kill_app,
                                 is_sub_proc=True)
    display = Display(cmd_args,face_detector_out_que,log_que,LOG_LEVEL,kill_app, is_sub_proc=True)

    display.start()
    face_detector.start()
    camera.start()

    sleep(60)

    camera.stop()
    face_detector.stop()
    display.stop()

    logger.info('Face Detector Pipeline test completed')
    que_listener.stop()


if __name__ == '__main__':
    test_face_detector_pipeline()