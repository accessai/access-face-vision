from time import time
import traceback
from queue import Empty

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from access_face_vision.access_logger import get_logger
from access_face_vision.component import AccessComponent
from access_face_vision.face_group_manager import FaceGroupLocalManager
from access_face_vision.utils import roundUp


class CosineSimFaceRecogniser(AccessComponent):

    def __init__(self, cmd_args, in_que, out_que, fg_manager, log_que, log_level, kill_app, is_sub_proc=False):
        self.is_sub_proc = is_sub_proc

        if is_sub_proc is False:
            self.fg_manager = fg_manager

        super(CosineSimFaceRecogniser, self).__init__(face_recogniser,
                                                      cmd_args=cmd_args,
                                                      in_que=in_que,
                                                      out_que=out_que,
                                                      log_que=log_que,
                                                      log_level=log_level,
                                                      kill_app=kill_app)

    def __call__(self, obj):
        fg = self.fg_manager.get_face_group(obj['face_group'])
        return recognise(obj, fg.embeddings, fg.labels, fg.faceIds, self.cmd_args.recognition_threshold)


def recognise(obj, trained_embeddings, trained_labels, trained_face_ids, recognition_threshold):
    time_in = time()

    detected_embeddings = obj.get('embeddings', [])
    if len(detected_embeddings) > 0 and len(trained_embeddings)>0 and len(trained_labels)>0:
        sims = cosine_similarity(trained_embeddings, detected_embeddings)
        indexes = np.argmax(sims, axis=0)
        confidence = np.max(sims, axis=0)
        detected_labels = trained_labels[indexes]
        detected_face_ids = trained_face_ids[indexes]

        for label, faceId, conf, face in zip(detected_labels, detected_face_ids, confidence, obj.get('detections')):
            face['label'] = label
            face['faceId'] = faceId
            face['recognition_confidence'] = conf
            if face.get('confidence'):
                face['confidence'] = roundUp((face['confidence'] + conf)/2.)
                conf = face['confidence']
            if conf < recognition_threshold:
                face['label'] = 'Unknown'



    time_out = time()
    obj['recognition_time'] = time_out - time_in

    return obj


def face_recogniser(cmd_args, in_que, out_que, log_que, log_level, kill_proc, kill_app):

    logger = get_logger(log_que, log_level)

    try:

        logger.debug("Loading Face recognition model...")
        fg_manager = FaceGroupLocalManager(cmd_args)
        fg = fg_manager.get_face_group(cmd_args.face_group)
        known_embeddings = fg.embeddings
        known_labels = fg.labels
        known_face_ids = fg.faceIds

        while kill_proc.value == 0 and kill_app.value == 0:

            try:
                obj = in_que.get(block=True, timeout=1)
            except Empty as emp:
                logger.debug('Input que to face encoder is empty.')
                continue

            if obj.get('done', False):
                out_que.put(obj)
                break

            obj = recognise(obj, known_embeddings, known_labels, known_face_ids, cmd_args.recognition_threshold)

            out_que.put(obj)

    except Exception as ex:
        logger.error(traceback.format_exc())
        kill_app.value = 1
    finally:
        logger.info('Exiting from Face Recogniser')