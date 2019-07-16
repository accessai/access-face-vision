import os
from access_face_vision import access_logger
from time import sleep, time
import traceback
from glob import glob

from PIL import Image
import numpy as np

from access_face_vision.component import AccessComponent
from access_face_vision.access_logger import get_logger


class ImageReader(AccessComponent):

    def __init__(self, cmd_args, out_que, log_que, log_level, kill_app, is_sub_proc=False):
        self.is_sub_proc = is_sub_proc

        super(ImageReader, self).__init__(image_reader,
                                           cmd_args=cmd_args,
                                           out_que=out_que,
                                           log_que=log_que,
                                           log_level=log_level,
                                           kill_app=kill_app)


def image_reader(cmd_args, out_que, log_que, log_level, kill_proc, kill_app):

    logger = get_logger(log_que, log_level)

    try:
        img_dir = cmd_args.img_dir
        images = glob(os.path.join(img_dir, "**/*.*"), recursive=True)
        logger.info("Reading images from path {}, Total files {}...".format(img_dir, len(images)))

        for img_path in images:

            if kill_proc.value > 0 or kill_app.value > 0:
                logger.warning('Exiting from ImageReader process')
                break

            if img_path.split(".")[-1].lower() not in ['jpg', 'jpeg']:
                logger.warning("Invalid image extension found. We support only jpeg/jpg images.")
                continue

            img = np.array(Image.open(img_path))
            obj = {"raw_frame": img, "cap_time": time(), 'factor': 1, 'small_rgb_frame': img,
                   'label': " ".join(os.path.basename(img_path).split(".")[0].split("_")[:-1])}
            out_que.put(obj)
        out_que.put({"done": True})
    except Exception as ex:
        logger.error(traceback.format_exc())
        kill_app.value = 1
    finally:
        logger.info('Exiting from Directory Reader')
