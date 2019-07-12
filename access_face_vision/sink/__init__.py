from time import time
import cv2


def draw_on_frame(obj):
    frame = obj.get('raw_frame')
    cap_time = obj.get('cap_time')
    out_time = time()
    total_time = out_time - cap_time

    for r in obj.get('detections', []):
        top_left, bottom_right = r['box']
        cv2.rectangle(img=frame, pt1=top_left, pt2=bottom_right, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, "{} {}".format(r.get('label', 'Unknown'), "{:0.2f}".format(r.get('confidence'))),
                    (top_left[0], top_left[1]-5), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8,
                    color=(0, 255, 0), thickness=1)

    for r in obj.get('uncertain_detection', []):
        top_left, bottom_right = r['box']
        cv2.rectangle(img=frame, pt1=top_left, pt2=bottom_right, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, "Unknown {}".format("{:0.2f}".format(r.get('confidence'))),
                    (top_left[0], top_left[1]), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8,
                    color=(0, 255, 0), thickness=1)

    time_txt = 'Detection: {:0.2f}, Encoding: {:0.2f}, Recog: {:0.2f} Total: {:0.2f}, FPS: {:0.2f}'.format(
        obj.get('detection_time',-1),
        obj.get('encoding_time',-1),
        obj.get('recognition_time',-1),
        total_time,
        1 / total_time
    )
    cv2.putText(frame, time_txt, (5, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 255, 0),
                thickness=1)

    return frame