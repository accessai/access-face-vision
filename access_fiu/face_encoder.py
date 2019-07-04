import logging
from access_fiu import access_logger
from time import sleep, time
from multiprocessing import Process
import traceback
from queue import Empty

from keras.models import load_model
import cv2
import numpy as np


class FaceEncoder(object):
    def __init__(self, in_que, out_que, quit, log_que, log_level=logging.INFO):
        self.quit = quit
        self.process = Process(target=encode_face, args=(in_que, out_que, quit, log_que, log_level))

    def __del__(self):
        self.quit.value = 1

    def start(self):
        self.process.daemon = True
        self.process.start()

    def stop(self):
        self.quit.value = 1
        self.process.close()
        self.process.join(timeout=1)


def encode_face(in_que, out_que, quit, log_que, log_level):

    access_logger.conf_worker_logger(log_que)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    try:
        logger.info("Loading model...")
        model = load_model('D:\models\\facenet_keras.h5')

        while quit.value != 1:

            try:
                obj = in_que.get(block=False)
                if obj.get('done', False):
                    out_que.put(obj)
                    break
                small_rgb_frame = obj.get('small_rgb_frame')
            except Empty as emp:
                logger.debug("Input que to encoder is empty.")
                sleep(0.1)
                continue

            time_in = time()
            faces = []
            for r in obj.get('detections', []):
                top_left, bottom_right = r['small_dims']
                x1, y1 = top_left
                x2, y2 = bottom_right

                face = small_rgb_frame[y1:y2, x1:x2].astype('float32')
                mean, std = face.mean(), face.std()
                face = (face-mean)/std
                face = cv2.resize(face, (160, 160))
                faces.append(face)
            if faces:
                faces = np.array(faces)
                embeds = model.predict(faces)
                obj['faces'] = faces
                obj['embeddings'] = embeds

            time_out = time()
            obj['encoding_time'] = time_out - time_in
            out_que.put(obj)

    except Exception as ex:
        logger.error(traceback.format_exc())
        quit.value = 1
    finally:
        logger.warning('Exiting from Face Encoder')