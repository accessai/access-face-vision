import logging
from access_face_vision import access_logger
from multiprocessing import Queue, Value

from time import sleep
import signal
import sys
from time import time
from PIL import Image
import numpy as np

from access_face_vision.source.camera import Camera
from access_face_vision.face_detector import FaceDetector
from access_face_vision.face_encoder import FaceEncoder
from access_face_vision.face_recogniser import CosineSimFaceRecogniser
from access_face_vision.sink.display import Display

from access_face_vision import utils
from access_face_vision import server


kill_app = Value('i', 0)


def signal_handler(signum, frame):
    logger.warning('Exiting from main process...')
    kill_app.value = 1
    que_listener.stop()
    sleep(1)
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class AccessFaceVisionImage(object):

    def __init__(self, cmd_args):
        self.cmd_args = cmd_args
        self.face_detector = FaceDetector(cmd_args, None, None, log_que, cmd_args.log, kill_app, is_sub_proc=False)
        self.face_encoder = FaceEncoder(cmd_args, None, None, log_que,cmd_args.log,kill_app, is_sub_proc=False)
        self.face_recogniser = CosineSimFaceRecogniser(cmd_args, None, None, log_que, cmd_args.log, kill_app, is_sub_proc=False)

    def parse_image(self, img_bytes):
        img = Image.open(img_bytes)

        if img.format not in ['jpg', 'jpeg']:
            img = img.convert('RGB')

        img = np.array(img)
        obj = self.face_detector({'cap_time': time(), 'raw_frame': img,
                           'small_rgb_frame': img, 'factor': 1.})
        obj = self.face_encoder(obj)
        obj = self.face_recogniser(obj)

        faces = []
        detections = obj.get('detections', [])
        detections.extend(obj.get('uncertain_detection', []))
        for r in detections:
            faces.append(
                {'faceId': 0, 'box': r['rectangular_coordinates'], 'label': r['label'], 'confidence': float(r['confidence'])}
            )

        return {'faces': faces}

    def parse_video(self, stream):
        pass

    def create_face_group(self, face_group_name):
        pass

    def append_to_face_group(self, img_bytes):
        pass

    def delete_face_group(self, face_group_name):
        pass

    def remove_from_face_group(self, face_id):
        pass

    def list_face_ids(self, face_group_name):
        pass


if __name__ == '__main__':

    cmd_args = utils.create_parser()

    logger, log_que, que_listener = access_logger.set_main_process_logger(cmd_args.log)
    afv = AccessFaceVisionImage(cmd_args)
    server_app = server.app
    server.setup_routes(cmd_args, afv)
    sleep(20)
    logger.info("Ready...")
    server_app.run(host=cmd_args.host, port=cmd_args.port)

