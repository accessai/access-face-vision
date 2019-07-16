import logging
import argparse
from multiprocessing import Queue, Value
from access_face_vision import access_logger
from time import sleep

log_que = Queue(-1)
que_listener = access_logger.get_listener_logger(log_que)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from access_face_vision.face_detector import FaceDetector
from access_face_vision.face_encoder import FaceEncoder
from access_face_vision.source.image_reader import ImageReader
from access_face_vision.embedding_generator import EmbeddingGenerator
from access_face_vision import utils


def train_face_recognition_model(cmd_args):

    kill_app = Value('i', 0)
    camera_out_que = Queue()
    detector_out_que = Queue()
    encoder_out_que = Queue()

    dir_reader = ImageReader(cmd_args, camera_out_que, log_que, 'info', kill_app, True)
    face_detector = FaceDetector(cmd_args, camera_out_que, detector_out_que, log_que, 'info', kill_app, True)
    face_encoder = FaceEncoder(cmd_args, detector_out_que, encoder_out_que, log_que, 'info', kill_app, True)
    embed_gen = EmbeddingGenerator(cmd_args, encoder_out_que, log_que, 'info', kill_app, True)

    face_detector.start()
    face_encoder.start()
    embed_gen.start()
    dir_reader.start()

    que_listener.start()
    while kill_app.value != 1:
        sleep(0.2)

    logger.info('Main process exited')
    que_listener.stop()


if __name__ == '__main__':
    cmd_args = utils.create_parser()
    train_face_recognition_model(cmd_args)
