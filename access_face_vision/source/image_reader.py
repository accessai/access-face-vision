import os
from access_face_vision import access_logger
from time import sleep, time
import traceback
from glob import glob

from PIL import Image
import numpy as np

from access_face_vision.component import AccessComponent
from access_face_vision import utils


class ImageReader(AccessComponent):

    def __init__(self, cmd_args, *args, **kwargs):
        kwargs['img_dir'] = cmd_args.img_dir
        super(ImageReader, self).__init__(read_from_directory, *args, **kwargs)


def read_from_directory(img_dir, out_que, log_que, log_level, kill_proc, kill_app):

    access_logger.conf_worker_logger(log_que)
    logger = utils.get_logger(log_que, log_level)

    try:
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
                   'label': os.path.basename(img_path).split(".")[0]}
            out_que.put(obj)
        out_que.put({"done": True})
    except Exception as ex:
        logger.error(traceback.format_exc())
    finally:
        logger.warning('Exiting from Directory Reader')
