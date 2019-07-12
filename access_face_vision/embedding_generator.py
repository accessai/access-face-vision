import logging
from access_face_vision import access_logger
from time import sleep, time
from multiprocessing import Process
import traceback
from queue import Empty

from numpy import savez_compressed


class EmbeddingGenerator(object):
    def __init__(self, in_que, quit, log_que, log_level=logging.INFO):
        self.quit = quit
        self.process = Process(target=generate_embeddings, args=(in_que, quit, log_que, log_level))

    def __del__(self):
        self.quit.value = 1

    def start(self):
        self.process.daemon = True
        self.process.start()

    def stop(self):
        self.quit.value = 1
        self.process.close()
        self.process.join(timeout=1)


def generate_embeddings(in_que, quit, log_que, log_level):
    access_logger.conf_worker_logger(log_que)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    try:
        logger.info("Generating embeddings...")

        embeddings, faces, labels = [], [], []
        while quit.value != 1:
            try:

                obj = in_que.get(block=False)
                if obj.get('done', False):
                    logger.info("Exiting from embedding generator")
                    break

            except Empty as emp:
                logger.debug("Input que to embedding generator is empty.")
                sleep(0.1)
                continue

            if len(obj.get('faces', [])) > 0:
                embeddings.extend(obj.get('embeddings',[]))
                faces.extend(obj.get('faces', []))
                labels.extend([obj.get('label')] * len(obj.get('faces')))

        assert len(embeddings) == len(faces) == len(labels), 'Emeds: {}, Faces: {}, Labels: {}'.format(len(embeddings), len(faces), len(labels))
        savez_compressed('../output/data2.npz', embeddings=embeddings, faces=faces, labels=labels)

    except Exception as ex:
        logger.error(traceback.format_exc())
        quit.value = 1
    finally:
        logger.warning('Exiting from Embedding Generator')
        quit.value = 1