from time import sleep, time
import traceback
from queue import Empty

from mtcnn.mtcnn import MTCNN
import tensorflow as tf

from access_face_vision.access_logger import get_logger
from access_face_vision.component import AccessComponent


class FaceDetector(AccessComponent):

    def __init__(self, cmd_args, in_que, out_que, log_que, log_level, kill_app, is_sub_proc=False):
        self.is_sub_proc = is_sub_proc

        if is_sub_proc is False:
            self.logger = get_logger(log_que, log_level)
            self.mtcnn = MTCNN()
            self.graph = tf.get_default_graph()
            self.session = tf.Session(graph=self.graph)

        super(FaceDetector, self).__init__(face_detector,
                                           cmd_args=cmd_args,
                                           in_que=in_que,
                                           out_que=out_que,
                                           log_que=log_que,
                                           log_level=log_level,
                                           kill_app=kill_app)

    def __call__(self, obj):
        with self.graph.as_default():
            with self.session.as_default():
                return detect_face(obj, self.mtcnn, self.cmd_args.detection_threshold,
                                   self.cmd_args.min_face_height, self.logger)


def detect_face(obj, mtcnn, detection_threshold, min_face_height, logger):
    time_in = time()

    red_frame_rgb = obj.get('small_rgb_frame', None)

    final_results, uncertain_results = [], []
    if red_frame_rgb is not None:
        factor = obj['factor']
        results = mtcnn.detect_faces(red_frame_rgb)

        for r in results:
            r['detection_confidence'] = r['confidence']
            if r['detection_confidence'] >= detection_threshold:
                x, y, width, height = r['box']
                s_x1, s_y1 = abs(x), abs(y)
                s_x2, s_y2 = s_x1 + width, s_y1 + height

                x, y = int(abs(x) * 1 / factor), int(abs(y) * 1 / factor)
                width = int(width * 1 / factor)
                height = int(height * 1 / factor)

                top_left = (x, y)
                bottom_right = (x + width, y + height)
                r['box'] = (top_left, bottom_right)
                r['small_dims'] = ((s_x1, s_y1), (s_x2, s_y2))
                r['label'] = 'Unknown'
                r['rectangular_coordinates'] = {'x1': int(x), 'x2': int(x + width), 'y1': int(y), 'y2': int(y + height)}
                if height < min_face_height:
                    logger.warning("Short face detected of size {}. Required {}".format(height, min_face_height))
                    uncertain_results.append(r)
                else:
                    final_results.append(r)
            else:
                logger.warning("Low face detection confidence {:0.2f}. Required {:0.2f}".format(
                    r.get('detection_confidence'), detection_threshold))

    results = final_results
    time_out = time()
    obj['detections'] = results
    obj['uncertain_detection'] = uncertain_results
    obj['detection_time'] = time_out - time_in

    return obj


def face_detector(cmd_args, in_que, out_que, log_que, log_level, kill_proc, kill_app):

    logger = get_logger(log_que, log_level)

    try:
        logger.debug('FaceDetector process starting')
        detection_threshold = cmd_args.detection_threshold
        min_face_height = cmd_args.min_face_height
        logger.info("Loading Face detector model...")
        mtcnn = MTCNN()

        while kill_proc.value == 0 and kill_app.value == 0:

            try:
                obj = in_que.get(block=True, timeout=1)
            except Empty as emp:
                logger.debug('Input que to face detector is empty.')
                continue

            if obj.get('done', False):
                out_que.put(obj)
                break

            obj = detect_face(obj,mtcnn,detection_threshold,min_face_height, logger)
            out_que.put(obj)

    except Exception as ex:
        logger.error(traceback.format_exc())
        kill_proc.value = 1
        kill_app.value = 1
    finally:
        logger.info('Exiting from Face Detector')
