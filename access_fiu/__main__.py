import logging
from access_fiu import access_logger
from multiprocessing import Queue, Value

# Setup logging
log_que = Queue(-1)
que_listener = access_logger.get_listener_logger(log_que)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from time import sleep
import signal
import sys

from access_fiu.camera import Camera
from access_fiu.display import Display
from access_fiu.face_detector import FaceDetector
from access_fiu.face_encoder import FaceEncoder
from access_fiu.face_recogniser import FaceRecogniser


quit = Value('i',0)


def signal_handler(signum, frame):
    logger.warning('Exiting from main process...')
    quit.value = 1
    que_listener.stop()
    sleep(1)
    sys.exit(0)


if __name__ == '__main__':

    camera_out_que = Queue()
    detector_out_que = Queue()
    encoder_out_que = Queue()
    recogniser_out_que = Queue()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    camera = Camera(camera_out_que, quit,{'device': 1},log_que, False)
    face_detector = FaceDetector(camera_out_que, detector_out_que, quit, log_que)
    face_encoder = FaceEncoder(detector_out_que, encoder_out_que, quit, log_que)
    face_recogniser = FaceRecogniser(encoder_out_que,recogniser_out_que, quit, log_que)
    display = Display(recogniser_out_que, quit, log_que)

    face_detector.start()
    face_encoder.start()
    face_recogniser.start()
    display.start()
    camera.start()

    que_listener.start()
    while quit.value != 1:
        sleep(0.2)

    que_listener.stop()
    logger.warning('Main process exited')
