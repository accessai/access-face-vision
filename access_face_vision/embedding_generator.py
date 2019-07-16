import traceback
from queue import Empty

from access_face_vision.access_logger import get_logger
from access_face_vision.component import AccessComponent
from access_face_vision.face_group_manager import FaceGroupLocalManager


class EmbeddingGenerator(AccessComponent):
    def __init__(self, cmd_args, in_que, log_que, log_level, kill_app, is_sub_proc=False):
        self.is_sub_proc = is_sub_proc

        super(EmbeddingGenerator, self).__init__(embedding_generator,
                                           cmd_args=cmd_args,
                                           in_que=in_que,
                                           log_que=log_que,
                                           log_level=log_level,
                                           kill_app=kill_app)


def embedding_generator(cmd_args, in_que, log_que, log_level, kill_proc, kill_app):

    logger = get_logger(log_que, log_level)

    try:
        logger.info("Generating embeddings...")
        face_group_manager = FaceGroupLocalManager(cmd_args)
        face_group_manager.create_face_group('default')

        while kill_proc.value == 0 and kill_app.value == 0:
            try:
                obj = in_que.get(block=True, timeout=1)
            except Empty as emp:
                logger.debug('Input que to face detector is empty.')
                continue

            if obj.get('done', False):
                kill_app.value = 1
                break

            if len(obj.get('faces', [])) == 1:
                face_group_manager.append_to_face_group(obj.get('face_group') or 'default',
                                                        obj['embeddings'][0],
                                                        obj['label'])
            else:
                logger.error("0 or more than 1 face detected for {}".format(obj['label']))

    except Exception as ex:
        logger.error(traceback.format_exc())
        kill_app.value = 1
    finally:
        logger.info('Exiting from Embedding Generator')