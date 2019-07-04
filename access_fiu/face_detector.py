from access_fiu import access_logger
import logging
from time import sleep, time
from multiprocessing import Process
import traceback

from mtcnn.mtcnn import MTCNN


class FaceDetector(object):
    def __init__(self, in_que, out_que, quit, log_que, log_level=logging.INFO):
        self.quit = quit
        self.process = Process(target=detect_face, args=(in_que, out_que, quit, log_que, log_level))

    def __del__(self):
        self.quit.value = 1

    def start(self):
        self.process.daemon = True
        self.process.start()

    def stop(self):
        self.quit.value = 1
        self.process.close()
        self.process.join(timeout=1)


def detect_face(in_que, out_que, quit, log_que, log_level):

    access_logger.conf_worker_logger(log_que)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    threshold = 0.8

    try:

        logger.debug("Loading Face detector model...")
        mtcnn = MTCNN()

        while quit.value != 1:

            if in_que.empty():
                sleep(0.1)
                logger.debug('Input que to face detector is empty.')
                continue

            time_in = time()
            obj = in_que.get(block=False)

            if obj.get('done', False):
                out_que.put(obj)
                break

            red_frame_rgb = obj['small_rgb_frame']
            factor = obj['factor']

            results = mtcnn.detect_faces(red_frame_rgb)

            final_results = []
            for r in results:
                if r.get('confidence',-1) >= threshold:
                    x, y, width, height = r['box']
                    s_x1, s_y1 = abs(x), abs(y)
                    s_x2, s_y2 = s_x1 + width, s_y1+height

                    x, y = int(abs(x) * 1/factor), int(abs(y) * 1/factor)
                    width = int(width * 1/factor)
                    height = int(height * 1/factor)

                    top_left = (x, y)
                    bottom_right = (x+width, y+height)
                    r['box'] = (top_left, bottom_right)
                    r['small_dims'] = ((s_x1, s_y1), (s_x2, s_y2))
                    final_results.append(r)

            results = final_results
            time_out = time()
            obj['detections'] = results
            obj['detection_time'] = time_out - time_in
            out_que.put(obj)
    except Exception as ex:
        logger.error(traceback.format_exc())
        quit.value = 1
    finally:
        logger.warning('Exiting from Face Detector')
