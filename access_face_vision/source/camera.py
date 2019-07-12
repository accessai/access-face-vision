import signal
from time import time, sleep
import math

import cv2

from access_face_vision import utils
from access_face_vision.component import AccessComponent
from access_face_vision.access_logger import get_logger


class Camera(AccessComponent):

    def __init__(self, cmd_args, out_queue, log_que, log_level, kill_app, draw_frames=False):
        super(Camera, self).__init__(capture,
                                     cmd_args=cmd_args,
                                     out_queue=out_queue,
                                     log_que=log_que,
                                     log_level=log_level,
                                     kill_app=kill_app,
                                     draw_frames=draw_frames)


def capture(cmd_args, out_queue, log_que, log_level, kill_proc, kill_app, draw_frames):

    logger = get_logger(log_que, log_level)

    device = cmd_args.camera_url if cmd_args.camera_url != '' else cmd_args.camera_index
    REQUIRED_FPS = cmd_args.fps
    CAMERA_WAIT = cmd_args.camera_wait
    img_dim = (cmd_args.img_width, cmd_args.img_height)
    NUM_FRAME_TO_SKIP = 2

    logger.info('Acquiring camera. Please wait...')
    sleep(CAMERA_WAIT)
    factor = cmd_args.img_red_factor
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    logger.info('Camera acquired')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_dim[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_dim[1])
    logger.info("Capturing Images with dimension: {}".format(img_dim))

    def exit_gracefully(signum, frame):
        kill_app.value = 1
        logger.warning('Terminating camera process due to kill signal')
        cap.release()
        cv2.destroyAllWindows()
        utils.clean_queue(out_queue)

    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    skip_count= 0
    tik = time()
    count=0

    if cap.isOpened():
        logger.info('Camera opened')
    else:
        logger.error('Unable to open camera')

    while cap.isOpened():
        if kill_proc.value > 0 or kill_app.value > 0:
            logger.warning('Breaking camera process loop')
            break

        ret, frame = cap.read()
        tok = time()
        count +=1

        if (tok-tik) > 2.0:
            tik = time()
            camera_fps = math.ceil(count / 2)
            NUM_FRAME_TO_SKIP = math.ceil((camera_fps - REQUIRED_FPS) / REQUIRED_FPS)
            logger.debug(str(('Camera FPS: ', camera_fps, ' Frames to skip: ',
                              NUM_FRAME_TO_SKIP, 'Effective FPS', REQUIRED_FPS)))
            count=0

        if ret is True:
            cv2.flip(frame, 1, frame)
            if skip_count >= NUM_FRAME_TO_SKIP:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                red_frame_rgb = cv2.resize(frame_rgb, (int(frame.shape[1] * factor), int(frame.shape[0] * factor)))
                out_queue.put({'cap_time': time(), 'raw_frame': frame,
                           'small_rgb_frame': red_frame_rgb, 'factor': factor}, block=True, timeout=5)
                skip_count=0

                if draw_frames:
                    logger.info("Required frame size {}. Captured size {}".format(img_dim, frame.shape))
                    cv2.imshow('CameraFeed', frame)

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            else:
                skip_count += 1

        else:
            break

    kill_app.value = 1
    cap.release()
    utils.clean_queue(out_queue)
    cv2.destroyAllWindows()
    logger.info('Exiting from Camera Process')
