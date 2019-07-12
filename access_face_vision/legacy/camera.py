import signal
import logging
from access_face_vision import access_logger
from multiprocessing import Value, Process
from time import time, sleep
import math
import argparse
import yaml

import cv2

from access_face_vision import utils


class Camera(object):

    def __init__(self, output_queue, quit, args, log_que, draw_frames=False, log_level=logging.INFO):
        self.quit = quit
        self.device = args.camera_index or args.camera_url
        self.img_dim = (args.img_width, args.img_height)
        self.process = Process(target=capture, args=(output_queue, quit, self.device, log_que, log_level, draw_frames,
                                                     self.img_dim))

    def __del__(self):
        self.quit.value = 1

    def start(self):
        self.process.daemon = True
        self.process.start()

    def stop(self):
        self.quit.value = 1
        self.process.close()
        self.process.join(timeout=1)


def capture(queue, quit, device, log_que, log_level, draw_frames=False, img_dim=(1280,720)):

    access_logger.conf_worker_logger(log_que)
    logger = utils.get_logger(log_que, log_level)

    NUM_FRAME_TO_SKIP = 2

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file n
    logger.info('Aquiring camera. Please wait...')
    sleep(25)
    factor = utils.REDUCTION_FACTOR
    # cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture('../output/bg3.mp4')
    logger.info('Camera acquired')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_dim[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_dim[1])
    logger.info("Capturing Images with dimension: {}".format(img_dim))

    def exit_gracefully(signum, frame):
        logger.warning('Terminating camera process due to kill signal')
        cap.release()
        cv2.destroyAllWindows()
        utils.clean_queue(queue)

    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    # Read until video is completed
    if queue is None:
        logger.error('No queue to put frame')
        quit.value = 1
        return

    skip_count= 0
    tik = time()
    count=0

    if cap.isOpened():
        logger.info('Camera opened')
    else:
        logger.error('Unable to open camera')

    while cap.isOpened():
        # Capture frame-by-frame
        if quit.value > 0:
            logger.warning('Exiting from camera process')
            break

        ret, frame = cap.read()
        tok = time()
        count +=1

        if (tok-tik) > 2.0:
            tik = time()
            camera_fps = math.ceil(count / 2)
            NUM_FRAME_TO_SKIP = math.ceil((camera_fps - utils.REQUIRED_FPS) / utils.REQUIRED_FPS)
            logger.debug(str(('Camera FPS: ', camera_fps, ' Frames to skip: ',
                              NUM_FRAME_TO_SKIP, 'Effective FPS', utils.REQUIRED_FPS)))
            count=0

        if ret is True:
            # cv2.flip(frame, 1, frame)
            if skip_count >= NUM_FRAME_TO_SKIP:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                red_frame_rgb = cv2.resize(frame_rgb, (int(frame.shape[1] * factor), int(frame.shape[0] * factor)))
                queue.put({'cap_time': time(), 'raw_frame': frame,
                           'small_rgb_frame': red_frame_rgb, 'factor': factor}, block=True, timeout=5)
                skip_count=0

                if draw_frames:
                    rtik = time()
                    img = cv2.resize(frame, (utils.RED_IMG_WIDTH, utils.RED_IMG_HEIGHT))
                    rtok = time()
                    logger.info("Resizing time: {}".format((rtok-rtik)))
                    logger.info("Required frame size {}. Captured size {}".format(img_dim, frame.shape))
                    cv2.imshow('Frame0', frame)

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.imwrite('tmp.jpg', frame)
                        quit.value = 1
                        break
            else:
                skip_count += 1

        # Break the loop
        else:
            break

    cap.release()
    # quit.value = 1
    utils.clean_queue(queue)
    cv2.destroyAllWindows()
    logger.warning('Exiting from Camera Process')


if __name__ == '__main__':
    from queue import Queue

    logging.basicConfig(format= access_logger.FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)


    def _create_parser():
        parser = argparse.ArgumentParser(description='Guest Magic Parser')
        parser.add_argument('--config',
                            type=str,
                            required=True,
                            help='configuration file')

        return parser.parse_args()

    args = _create_parser()
    config_path = args.config

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = int(input('enter camera index: '))
    conf = {'device': device}
    capture(Queue(), Value('i', 0), device, True,
            img_dim=(config.get('capture_img_width', 1280), config.get('capture_img_height', 720)))
