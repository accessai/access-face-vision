import os
from time import time
import traceback
from queue import Empty

from keras.models import load_model
import cv2
import numpy as np

from access_face_vision.access_logger import get_logger
from access_face_vision.component import AccessComponent


class FaceEncoder(AccessComponent):

    def __init__(self, cmd_args, in_que, out_que, log_que, log_level, kill_app, is_sub_proc=False):
        self.is_sub_proc = is_sub_proc

        if is_sub_proc is False:
            self.face_encoder_model = load_model('./models/accessai_v1_facesim_weights.h5')
            self.face_encoder_model._make_predict_function()

        super(FaceEncoder, self).__init__(face_encoder,
                                          cmd_args=cmd_args,
                                          in_que=in_que,
                                          out_que=out_que,
                                          log_que=log_que,
                                          log_level=log_level,
                                          kill_app=kill_app)

    def __call__(self, obj):
        return encode_face(self.face_encoder_model, obj)


def encode_face(face_encoder_model, obj):
    time_in = time()
    small_rgb_frame = obj.get('small_rgb_frame')

    faces = []
    for r in obj.get('detections', []):
        top_left, bottom_right = r['small_dims']
        x1, y1 = top_left
        x2, y2 = bottom_right

        face = small_rgb_frame[y1:y2, x1:x2].astype('float')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        face = cv2.resize(face, (160, 160))
        faces.append(face)
    if faces:
        faces = np.array(faces)
        embeds = face_encoder_model.predict(faces)
        obj['faces'] = faces
        obj['embeddings'] = embeds

    time_out = time()
    obj['encoding_time'] = time_out - time_in

    return obj


def face_encoder(cmd_args, in_que, out_que, log_que, log_level, kill_proc, kill_app):

    logger = get_logger(log_que, log_level)

    try:
        logger.debug('Face Encoder process starting')
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'accessai_v1_facesim_weights.h5')
        face_encoder_model = load_model(model_path)
        logger.info("face encoder model loaded")

        while kill_proc.value == 0 and kill_app.value == 0:

            try:
                obj = in_que.get(block=True, timeout=1)
            except Empty as emp:
                logger.debug('Input que to face encoder is empty.')
                continue

            if obj.get('done', False):
                out_que.put(obj)
                break

            obj = encode_face(face_encoder_model, obj)

            out_que.put(obj)

    except Exception as ex:
        logger.error(traceback.format_exc())
        kill_app.value = 1
    finally:
        logger.info('Exiting from Face Encoder')