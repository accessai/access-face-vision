from multiprocessing import Queue, Value
from time import sleep
import signal
import sys

from access_fiu.camera import Camera
from access_fiu.display import Display
from access_fiu.face_detector import FaceDetector

quit = Value('i',0)


def signal_handler(signum, frame):
    print('Exiting from main process...')
    quit.value = 1
    sleep(1)
    sys.exit(0)


if __name__ == '__main__':

    camera_put_que = Queue()
    display_get_que = Queue()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    camera = Camera(camera_put_que, quit,{},False)
    image_processor = FaceDetector(camera_put_que, display_get_que, quit)
    display = Display(display_get_que, quit)

    camera.start()
    image_processor.start()
    display.start()

    while quit.value != 1:
        sleep(0.2)

    print('Main process exited')
