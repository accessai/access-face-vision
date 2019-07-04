import math

import cv2
import numpy as np


def draw_round_rectangle(img, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    x1 -= 20
    y1 -= 20
    x2 -= 20
    y2 -= 20

    width = x2 - x1
    height = y2 - y1
    r = int(height * 0.02 + 1)

    border_radius = math.ceil(abs(height-width)*0.1)
    line_thickness = 2
    edge_shift = int(line_thickness/2.0)
    color = (255,255,255)

    cv2.rectangle(img,(x1,y1), (x2,y2),color,thickness=-1, lineType=cv2.LINE_AA)

    # #draw lines
    # #top
    # cv2.line(img, (border_radius, edge_shift),
    # (width - border_radius, edge_shift), color, line_thickness)
    # #bottom
    # cv2.line(img, (border_radius, height-line_thickness),
    # (width - border_radius, height-line_thickness), color, line_thickness)
    # #left
    # cv2.line(img, (edge_shift, border_radius),
    # (edge_shift, height  - border_radius), color, line_thickness)
    # #right
    # cv2.line(img, (width - line_thickness, border_radius),
    # (width - line_thickness, height  - border_radius), color, line_thickness)

    # corners
    cv2.ellipse(img, (x1 + r, y1 + r),
                (border_radius, border_radius), 180, 0, 100, color, line_thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x2 - (border_radius + line_thickness), y1),
                (border_radius, border_radius), 270, 0, 90, color, line_thickness)
    cv2.ellipse(img, (x2 - (border_radius + line_thickness), y2 - (border_radius + line_thickness)),
                (border_radius, border_radius), 10, 0, 90, color, line_thickness)
    cv2.ellipse(img, (x1 + edge_shift, y2 - (border_radius + line_thickness)),
                (border_radius, border_radius), 90, 0, 90, color, line_thickness)


def draw_rect_polly(img, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    x1 -= 20
    y1 -= 20
    x2 -= 20
    y2 -= 20

    width = x2 - x1
    height = y2 - y1
    r = int(height * 0.02 + 1)

    border_radius = math.ceil(abs(height - width) * 0.1)
    line_thickness = 2
    edge_shift = int(line_thickness / 2.0)
    color = (255, 255, 255)

    pts = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], dtype=np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts], True, color, thickness=5, lineType=cv2.LINE_AA)
    cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)


def draw_bounding_box(top_left, bottom_right, overlay, color, inner):

    x1,y1 = top_left
    x2, y2 = bottom_right
    width = x2-x1
    height = y2-y1

    s = int(width * 0.3)

    if inner:
        thick =1
    else:
        r = int(height*0.02+1)
        x1 = max(0,x1-r)
        y1 = max(0, y1-r)
        x2 = min(overlay.shape[1], x2+r)
        y2 = min(overlay.shape[0], y2+r)
        thick=2

    cv2.line(overlay, (x1,y1), (x1+s, y1), color=color,thickness=thick,lineType=cv2.LINE_AA)
    cv2.line(overlay, (x1,y1), (x1, y1+s), color=color,thickness=thick,lineType=cv2.LINE_AA)

    cv2.line(overlay, (x2, y1), (x2 - s, y1), color=color, thickness=thick, lineType=cv2.LINE_AA)
    cv2.line(overlay, (x2, y1), (x2, y1 + s), color=color, thickness=thick, lineType=cv2.LINE_AA)

    cv2.line(overlay, (x2, y2), (x2 - s, y2), color=color, thickness=thick, lineType=cv2.LINE_AA)
    cv2.line(overlay, (x2, y2), (x2, y2 - s), color=color, thickness=thick, lineType=cv2.LINE_AA)

    cv2.line(overlay, (x1, y2), (x1 + s, y2), color=color, thickness=thick, lineType=cv2.LINE_AA)
    cv2.line(overlay, (x1, y2), (x1, y2 - s), color=color, thickness=thick, lineType=cv2.LINE_AA)


def draw_text_rectangle(img, top_left, bottom_right, fill_color, border_color):

    x1, y1 = top_left
    x2, y2 = bottom_right
    width = x2 - x1
    height = y2 - y1

    r = int(height * 0.02 + 1)
    ox1 = max(0, x1 - r)
    oy1 = max(0, y1 - r)
    ox2 = min(img.shape[1], x2 + r)
    oy2 = min(img.shape[0], y2 + r)

    cv2.rectangle(img, (x1,y1), (x2,y2), fill_color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (ox1,oy1), (ox2,oy2), border_color, thickness=1, lineType=cv2.LINE_AA)
