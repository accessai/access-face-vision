import logging
from access_fiu import access_logger
from multiprocessing import Process
from time import sleep, time

import cv2


class Display(object):

    def __init__(self, in_que, quit, log_que, log_level=logging.INFO):
        self.quit = quit
        self.process = Process(target=display, args=(in_que, quit, log_que, log_level))

    def __del__(self):
        self.quit.value = 1

    def start(self):
        self.process.daemon = True
        self.process.start()

    def stop(self):
        self.quit.value = 1
        self.process.close()
        self.process.join(timeout=1)


def display(in_que, quit, log_que, log_level):

    access_logger.conf_worker_logger(log_que)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    try:
        # throw out old frames
        while not in_que.empty():
            in_que.get()

        while quit.value != 1:

            if in_que.empty():
                sleep(0.1)
                continue

            results = in_que.get(block=False)
            frame = results.get('raw_frame')
            cap_time = results.get('cap_time')
            out_time = time()
            total_time = out_time - cap_time

            for r in results.get('detections', []):
                top_left, bottom_right = r['box']
                cv2.rectangle(img=frame, pt1=top_left, pt2=bottom_right, color=(0,255,0), thickness=2)
                cv2.putText(frame, "{} {}".format(r.get('label'), "{:0.2f}".format(r.get('confidence'))), (top_left[0], top_left[1]), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8,
                            color=(0, 255, 0), thickness=1)

            time_txt = 'Detection: {:0.2f}, Encoding: {:0.2f}, Recog: {:0.2f} Total: {:0.2f}, FPS: {:0.2f}'.format(
                results.get('detection_time'),
                results.get('encoding_time'),
                results.get('recognition_time'),
                total_time,
                1/total_time
            )
            cv2.putText(frame,time_txt, (5,30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0,255,0), thickness=1)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                quit.value = 1
                break
                pass
    finally:
        print('Exiting from Display process')
        quit.value = 1
