import os
import logging
from access_fiu import access_logger
from time import sleep, time
from multiprocessing import Process
import traceback
from glob import glob

from PIL import Image
import numpy as np


class DirectoryReader(object):
    def __init__(self, img_dir, out_que, quit, log_que, log_level=logging.INFO):
        self.quit = quit
        self.process = Process(target=read_from_directory, args=(img_dir, out_que, quit, log_que, log_level))

    def __del__(self):
        self.quit.value = 1

    def start(self):
        self.process.daemon = True
        self.process.start()

    def stop(self):
        self.quit.value = 1
        self.process.close()
        self.process.join(timeout=1)


def read_from_directory(img_dir, out_que, quit, log_que, log_level):
    access_logger.conf_worker_logger(log_que)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    try:
        images = glob(os.path.join(img_dir, "**/*.*"), recursive=True)
        logger.info("Reading images from path {}, Total files {}...".format(img_dir, len(images)))

        for img_path in images:

            if img_path.split(".")[-1].lower() not in ['jpg', 'jpeg']:
                logger.warning("Invalid image extension found. We support only jpeg/jpg images.")
                continue

            img = np.array(Image.open(img_path))
            obj = {"raw_frame": img, "cap_time": time(), 'factor': 1, 'small_rgb_frame': img,
                   'label': os.path.basename(img_path).split(".")[0]}
            out_que.put(obj)
        out_que.put({"done": True})
    except Exception as ex:
        logger.error(traceback.format_exc())
        quit.value = 1
    finally:
        logger.warning('Exiting from Directory Reader')
