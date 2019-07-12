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


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', help='Path to image to process', required=True, type=str)

    return parser.parse_args()


def generate_embeddings(args):

    quit = Value('i', 0)
    camera_out_que = Queue()
    detector_out_que = Queue()
    encoder_out_que = Queue()

    dir_reader = ImageReader(args.image_dir, camera_out_que, quit, log_que)
    face_detector = FaceDetector(camera_out_que, detector_out_que, quit, log_que)
    face_encoder = FaceEncoder(detector_out_que, encoder_out_que, quit, log_que)
    embed_gen = EmbeddingGenerator(encoder_out_que, quit, log_que)

    face_detector.start()
    face_encoder.start()
    embed_gen.start()
    dir_reader.start()

    que_listener.start()
    while quit.value != 1:
        sleep(0.2)

    que_listener.stop()
    logger.warning('Main process exited')


if __name__ == '__main__':
    args = _create_parser()
    generate_embeddings(args)
