from utils.utils import preproc, vis
from utils.utils import BaseEngine
import numpy as np
import cv2
import time
import os
import argparse

class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 80  # your model classes

if __name__ == '__main__':
    """
    sample:
    - camera: python3 trt.py -e yolov7.trt --end2end --camera /dev/video0
    - video: python3 trt.py -e yolov7.trt --end2end -v /home/deal/Downloads/trafic_example.mp4
    - img: python3 trt.py -e yolov7.trt  -i src/1.jpg -o yolov8n-1.jpg
    Press q to exit.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("-i", "--image", help="image path")
    parser.add_argument("-o", "--output", help="image output path")
    parser.add_argument("-v", "--video",  help="video path")
    parser.add_argument("--camera", help="use input from usb camera")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="use end2end engine")

    args = parser.parse_args()
    print(args)

    pred = Predictor(engine_path=args.engine)
    pred.get_fps()
    img_path = args.image
    video = args.video
    camera = args.camera

    # prioritize camera
    if camera:
      video = None
    
    if img_path:
      origin_img = pred.inference(img_path, conf=0.1, end2end=args.end2end)

      cv2.imwrite("%s" %args.output , origin_img)
    if video or camera:
      pred.detect_video(video, camera, conf=0.1, end2end=args.end2end)