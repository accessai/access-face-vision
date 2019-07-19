import pkg_resources
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disables Tensorflow warning messages

from access_face_vision import access_logger

from time import sleep
import signal
import sys
from time import time
from multiprocessing import Queue, Value

import numpy as np

from access_face_vision.source.camera import Camera
from access_face_vision.face_detector import FaceDetector
from access_face_vision.face_encoder import FaceEncoder
from access_face_vision.face_recogniser import CosineSimFaceRecogniser
from access_face_vision.sink.display import Display
from access_face_vision.face_group_manager import FaceGroupLocalManager
from access_face_vision.train_face_recognition_model import train_face_recognition_model
from access_face_vision import utils
from access_face_vision import server
from access_face_vision.exceptions import AccessException

kill_app = Value('i', 0)


def signal_handler(signum, frame):
    logger.warning('Exiting from main process...')
    kill_app.value = 1

    if server_app:
        server_app.stop()

    que_listener.stop()
    sleep(1)
    sys.exit(0)


class AccessFaceVision(object):

    def __init__(self, cmd_args):
        self.cmd_args = cmd_args
        self.fg_manager = FaceGroupLocalManager(cmd_args)


class AccessFaceVisionImage(AccessFaceVision):
    def __init__(self, cmd_args):
        super(AccessFaceVisionImage, self).__init__(cmd_args)
        self.face_detector = FaceDetector(cmd_args, None, None, log_que, cmd_args.log, kill_app, is_sub_proc=False)
        self.face_encoder = FaceEncoder(cmd_args, None, None, log_que,cmd_args.log,kill_app, is_sub_proc=False)
        self.face_recogniser = CosineSimFaceRecogniser(cmd_args, None, None, self.fg_manager,
                                                       log_que, cmd_args.log, kill_app, is_sub_proc=False)

    def parse_image(self, img, face_group):

        img = np.array(img)
        obj = self.face_detector({'cap_time': time(), 'raw_frame': img,
                           'small_rgb_frame': img, 'factor': 1.,'face_group': face_group})
        obj = self.face_encoder(obj)
        obj = self.face_recogniser(obj)

        faces = []
        detections = obj.get('detections', [])
        detections.extend(obj.get('uncertain_detection', []))
        for face in detections:
            faces.append(
                {'faceId': face['faceId'], 'box': face['rectangular_coordinates'], 'label': face['label'],
                 'confidence': float(face['confidence'])}
            )

        return {'faces': faces}

    def encode(self, img):

        obj = self.face_detector({'cap_time': time(), 'raw_frame': img,
                                  'small_rgb_frame': img, 'factor': 1.})
        obj = self.face_encoder(obj)

        return obj

    def create_face_group(self, fg_name):
        self.fg_manager.create_face_group(fg_name)
        return {'message': 'Face group {} created successfully'.format(fg_name)}

    def append_to_face_group(self, fg_name, img, label):
        img = np.array(img)
        obj = self.encode(img)
        faces = obj.get('detections', [])

        if len(faces) == 0:
            return AccessException('No face found', error_code=400)
        elif len(faces) > 1:
            return AccessException('More than one face found', error_code=400)
        else:
            embedding = obj['embeddings'][0]
            face_id = self.fg_manager.append_to_face_group(fg_name, embedding, label)
            return {'face_id': face_id}

    def delete_face_group(self, fg_name):
        self.fg_manager.delete_face_group(fg_name)
        return {"success": "{} deleted".format(fg_name)}

    def delete_from_face_group(self, fg_name, face_id):
        self.fg_manager.delete_from_face_group(face_id, fg_name)

        return {"success": "{} deleted from {}".format(face_id, fg_name)}

    def list_face_ids(self, fg_name):
        fg = self.fg_manager.get_face_group(fg_name)

        return {'face_ids': fg.faceIds.tolist()}


class AccessFaceVisionVideo(AccessFaceVision):

    def __init__(self, cmd_args):
        super(AccessFaceVisionVideo, self).__init__(cmd_args)
        self.camera_out_que = Queue()
        self.face_detector_out_que = Queue()
        self.face_encoder_out_que = Queue()
        self.face_recogniser_out_que = Queue()

        self.camera = Camera(cmd_args, self.camera_out_que, log_que, cmd_args.log, kill_app, draw_frames=False)
        self.face_detector = FaceDetector(cmd_args, self.camera_out_que, self.face_detector_out_que, log_que, cmd_args.log, kill_app, is_sub_proc=True)
        self.face_encoder = FaceEncoder(cmd_args, self.face_detector_out_que, self.face_encoder_out_que, log_que, cmd_args.log,kill_app, is_sub_proc=True)
        self.face_recogniser = CosineSimFaceRecogniser(cmd_args, self.face_encoder_out_que,
                                                       self.face_recogniser_out_que, None, log_que, cmd_args.log, kill_app, is_sub_proc=True)
        self.display = Display(cmd_args,self.face_recogniser_out_que, log_que, cmd_args.log, kill_app, is_sub_proc=True)

    def start(self):
        self.display.start()
        self.face_recogniser.start()
        self.face_encoder.start()
        self.face_detector.start()
        self.camera.start()

    def stop(self):
        self.camera.stop()
        self.face_detector.stop()
        self.face_encoder.stop()
        self.face_recogniser.stop()
        self.display.stop()


if __name__ == '__main__':

    cmd_args = utils.create_parser()

    if cmd_args.face_group:
        cmd_args.face_group = os.path.basename(cmd_args.face_group)
        cmd_args.face_group_dir = os.path.dirname(cmd_args.face_group) or './'

    logger, log_que, que_listener = access_logger.set_main_process_logger(cmd_args.log,
                                                                          cmd_args.log_screen,
                                                                          cmd_args.log_file)

    model_file = pkg_resources.resource_filename(__name__, os.path.join('models', 'accessai_v1_facesim_weights.h5'))
    utils.get_file("https://storage.googleapis.com/accessai/accessai_v1_facesim_weights.h5", model_file)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    server_app = None
    if cmd_args.mode == 'server':
        afv = AccessFaceVisionImage(cmd_args)
        server_app = server.app
        server.setup_routes(cmd_args, afv, logger)
        logger.info("Ready...")
        server_app.run(host=cmd_args.host, port=cmd_args.port,
                       register_sys_signals=False, access_log=False)
    elif cmd_args.mode == 'live-video':
        afv = AccessFaceVisionVideo(cmd_args)
        afv.start()

        while kill_app.value == 0:
            sleep(0.2)

        afv.stop()
        logger.info("Exiting application")
        que_listener.stop()

    elif cmd_args.mode == "train":
        train_face_recognition_model(cmd_args, logger, log_que)
        logger.info("Exiting application")
        que_listener.stop()
    else:
        raise RuntimeError('Unknown mode.')

