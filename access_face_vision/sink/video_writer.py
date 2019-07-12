from queue import Empty
import traceback

import cv2

from access_face_vision.access_logger import get_logger
from access_face_vision.component import AccessComponent
from access_face_vision.sink import draw_on_frame


class VideoWriter(AccessComponent):

    def __init__(self, cmd_args, in_que, log_que, log_level, kill_app):

        super(VideoWriter, self).__init__(video_writer,
                                          cmd_args=cmd_args,
                                          in_que=in_que,
                                          log_que=log_que,
                                          log_level=log_level,
                                          kill_app=kill_app)


def video_writer(cmd_args, in_que, log_que, log_level, kill_proc, kill_app):

    logger = get_logger(log_que, log_level)

    video = cv2.VideoWriter(cmd_args.video_out,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   cmd_args.fps ,
                                   (cmd_args.img_width,cmd_args.img_height))

    try:

        while kill_app.value == 0 and kill_proc.value == 0:

            try:
                obj = in_que.get(block=True, timeout=1)
            except Empty as emp:
                logger.debug('Input que to display is empty.')
                continue

            if obj.get('done', False):
                break

            frame = draw_on_frame(obj)
            video.write(frame)

    except Exception as ex:
        logger.error(traceback.format_exc())
        kill_app.value = 1
    finally:
        logger.info('Exiting from VideoWriter process')
