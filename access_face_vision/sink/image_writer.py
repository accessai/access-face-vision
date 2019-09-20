import os
from queue import Empty
import traceback
from time import time

import cv2

from access_face_vision.access_logger import get_logger
from access_face_vision.component import AccessComponent
from access_face_vision.sink import draw_on_frame


class ImageWriter(AccessComponent):

    def __init__(self, cmd_args, in_que, log_que, log_level, kill_app, is_sub_proc=False):
        self.is_sub_proc = is_sub_proc
        self.logger = get_logger(log_que, log_level)

        super(ImageWriter, self).__init__(None,
                                      cmd_args=cmd_args,
                                      in_que=in_que,
                                      log_que=log_que,
                                      log_level=log_level,
                                      kill_app=kill_app)

    def __call__(self, obj):
        frame = obj.get('raw_frame')
        thickness = int(frame.shape[0]/1024 * 1.7)
        fontscale = frame.shape[0]/1024 * 0.9

        for r in obj.get('faces', []):
            if r.get('label') != 'Unknown':
                label = "{} {}".format(r.get('label', 'Unknown'), "{:0.2f}".format(r.get('confidence')))
            else:
                label= ""

            top_left, bottom_right = (r['box']['x1'], r['box']['y1']), (r['box']['x2'], r['box']['y2'])
            self.draw_inner_bounding_box(top_left, bottom_right,frame, (0, 255, 0), max(1,int(thickness-1.5)), inner=True)
            self.draw_inner_bounding_box(top_left, bottom_right,frame, (0, 255, 0), thickness, inner=False)
            # cv2.rectangle(img=frame, pt1=top_left, pt2=bottom_right, color=(0, 255, 0), thickness=thickness)
            # cv2.putText(frame, label,
            #             (top_left[0], top_left[1] - 15), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontscale,
            #             color=(0, 255, 0), thickness=thickness)

        image_path = os.path.join(self.cmd_args.save_dir, os.path.basename(self.cmd_args.img_in))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, frame)

        return image_path

    def draw_inner_bounding_box(self, top_left, bottom_right, overlay, color, thick, inner=True):

        x1, y1 = top_left
        x2, y2 = bottom_right
        width = x2 - x1
        height = y2 - y1

        s = int(width * 0.4)

        if not inner:
            r = int(height * 0.06 + 1)
            x1 = max(0, x1 - r)
            y1 = max(0, y1 - r)
            x2 = min(overlay.shape[1], x2 + r)
            y2 = min(overlay.shape[0], y2 + r)

        cv2.line(overlay, (x1, y1), (x1 + s, y1), color=color, thickness=thick, lineType=cv2.LINE_AA)
        cv2.line(overlay, (x1, y1), (x1, y1 + s), color=color, thickness=thick, lineType=cv2.LINE_AA)

        cv2.line(overlay, (x2, y1), (x2 - s, y1), color=color, thickness=thick, lineType=cv2.LINE_AA)
        cv2.line(overlay, (x2, y1), (x2, y1 + s), color=color, thickness=thick, lineType=cv2.LINE_AA)

        cv2.line(overlay, (x2, y2), (x2 - s, y2), color=color, thickness=thick, lineType=cv2.LINE_AA)
        cv2.line(overlay, (x2, y2), (x2, y2 - s), color=color, thickness=thick, lineType=cv2.LINE_AA)

        cv2.line(overlay, (x1, y2), (x1 + s, y2), color=color, thickness=thick, lineType=cv2.LINE_AA)
        cv2.line(overlay, (x1, y2), (x1, y2 - s), color=color, thickness=thick, lineType=cv2.LINE_AA)