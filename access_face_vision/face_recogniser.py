from time import time
import traceback
from queue import Empty

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from access_face_vision.access_logger import get_logger
from access_face_vision.component import AccessComponent


class CosineSimFaceRecogniser(AccessComponent):

    def __init__(self, cmd_args, in_que, out_que, log_que, log_level, kill_app, is_sub_proc=False):
        self.is_sub_proc = is_sub_proc

        if is_sub_proc is False:
            encoded_faces = np.load(cmd_args.encoded_faces)
            self.known_embeddings = encoded_faces['embeddings']
            self.known_labels = encoded_faces['labels']

        super(CosineSimFaceRecogniser, self).__init__(face_recogniser,
                                                      cmd_args=cmd_args,
                                                      in_que=in_que,
                                                      out_que=out_que,
                                                      log_que=log_que,
                                                      log_level=log_level,
                                                      kill_app=kill_app)

    def __call__(self, obj):
        return recognise(obj, self.known_embeddings, self.known_labels)


def recognise(obj, trained_embeddings, trained_labels):
    time_in = time()

    detected_embeddings = obj.get('embeddings', [])
    if len(detected_embeddings) > 0:
        sims = cosine_similarity(trained_embeddings, detected_embeddings)
        indexes = np.argmax(sims, axis=0)
        confidence = np.max(sims, axis=0)
        detected_labels = trained_labels[indexes]

        for label, conf, face in zip(detected_labels, confidence, obj.get('detections')):
            face['label'] = " ".join(label.split("_")[:2])
            face['confidence'] = conf
            if conf < 0.65:
                face['label'] = 'Unknown'

    time_out = time()
    obj['recognition_time'] = time_out - time_in

    return obj


def face_recogniser(cmd_args, in_que, out_que, log_que, log_level, kill_proc, kill_app):

    logger = get_logger(log_que, log_level)

    try:

        logger.debug("Loading Face recognition model...")
        encoded_faces = np.load(cmd_args.encoded_faces)
        known_embeddings = encoded_faces['embeddings']
        known_labels = encoded_faces['labels']

        while kill_proc.value == 0 and kill_app.value == 0:

            try:
                obj = in_que.get(block=True, timeout=1)
            except Empty as emp:
                logger.debug('Input que to face encoder is empty.')
                continue

            if obj.get('done', False):
                out_que.put(obj)
                break

            obj = recognise(obj, known_embeddings, known_labels)

            out_que.put(obj)

    except Exception as ex:
        logger.error(traceback.format_exc())
        kill_app.value = 1
    finally:
        logger.info('Exiting from Face Recogniser')