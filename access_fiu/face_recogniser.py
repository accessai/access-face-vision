from access_fiu import access_logger
import logging
from time import sleep, time
from multiprocessing import Process
import traceback

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class FaceRecogniser(object):
    def __init__(self, in_que, out_que, quit, log_que, log_level=logging.INFO):
        self.quit = quit
        self.process = Process(target=recognise_face, args=(in_que, out_que, quit, log_que, log_level))

    def __del__(self):
        self.quit.value = 1

    def start(self):
        self.process.daemon = True
        self.process.start()

    def stop(self):
        self.quit.value = 1
        self.process.close()
        self.process.join(timeout=1)


def recognise_face(in_que, out_que, quit, log_que, log_level):

    access_logger.conf_worker_logger(log_que)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    try:

        logger.debug("Loading Face recognition model...")
        data = np.load('../output/data.npz')
        trained_embeddings = data['embeddings']
        trained_labels = data['labels']
        assert trained_labels.shape[0] == trained_embeddings.shape[0], 'Trained labels and embeddings shape differs.'

        while quit.value != 1:

            if in_que.empty():
                sleep(0.1)
                logger.debug('Input que to face recogniser is empty.')
                continue

            time_in = time()
            obj = in_que.get(block=False)

            if obj.get('done', False):
                out_que.put(obj)
                break

            detected_embeddings = obj.get('embeddings', [])
            if len(detected_embeddings) > 0:
                sims = cosine_similarity(trained_embeddings, detected_embeddings)
                indexes = np.argmax(sims, axis=0)
                confidence = np.max(sims, axis=0)
                detected_labels = trained_labels[indexes]

                for label, conf, face in zip(detected_labels, confidence, obj.get('detections')):
                    face['label'] = label
                    face['confidence'] = conf

            time_out = time()
            obj['recognition_time'] = time_out-time_in
            out_que.put(obj)
    except IndexError as ie:
        logger.error(traceback.format_exc())
        quit.value=1
    except Exception as ex:
        logger.error(traceback.format_exc())
        quit.value = 1
    finally:
        logger.warning('Exiting from Face Recogniser')