import os
import argparse
import math

import wget

def clean_queue(queue):

    try:
        while not queue.empty():
            queue.get(block=False)
    except Exception as ex:
        pass
    finally:
        return True


def get_file(url, out_file_path):
    if not os.path.exists(out_file_path):
        ensure_directory(os.path.dirname(out_file_path))
        wget.download(url, out_file_path)


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def roundUp(num):
    return math.ceil(num*100)/100.


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='Mode of application', required=False, type=str,
                        choices=['server', 'live-video', 'recorded-video', 'train'], default='server')
    parser.add_argument('--host', help='Host address to bind to', required=False, type=str, default='localhost')
    parser.add_argument('--port', help='Port number to bind to', required=False, type=int, default=5001)
    parser.add_argument('--cors', help='Cross Origin Resource Sharing host names', required=False, type=str, default='localhost')
    parser.add_argument('--camera-url', help='RTP camera URL on the network', required=False, type=str, default='')
    parser.add_argument('--camera-index', help='Camera Index', required=False, type=int, default=0)
    parser.add_argument('--camera_wait', help='Time to wait before capturing', type=int, required=False, default=25)
    parser.add_argument('--fps', help='FPS with which to capture frames', type=int, required=False, default=3)
    parser.add_argument('--video_out', help='Path of output video file', required=False, default='../output/video.mp4')
    parser.add_argument('--img_height', help='Image height to capture from camera', required=False, type=int, default=720)
    parser.add_argument('--img_width', help='Image width to capture from camera', required=False, type=int, default=1280)
    parser.add_argument('--face_group', help='Path to encoded faces file', required=False, type=str, default=None)
    parser.add_argument('--face_group_dir', help='Path to Face Group directory', required=False, type=str, default=None)
    parser.add_argument('--img_dir', help='Image directory', required=False, type=str, default='')
    parser.add_argument('--video_dir', help='Video directory', required=False, type=str, default='')
    parser.add_argument('--img_red_factor', help='Reduce Image size with this factor before processing', required=False, type=float, default=0.6)
    parser.add_argument('--detection_threshold', help='Face Detection threshold', required=False, type=float, default=0.8)
    parser.add_argument('--recognition_threshold', help='Face Detection threshold', required=False, type=float, default=0.7)
    parser.add_argument('--min_face_height', help='Minimum face height required for recognition', required=False, type=int, default=100)
    parser.add_argument('--log', help='Logging level', required=False, type=str, default='info')
    parser.add_argument('--log_screen', help='Print logging messages on console screen', required=False, type=bool, default=True)
    parser.add_argument('--log_file', help='Write log messages to this file', required=False, type=str, default=None)

    return parser.parse_args()

