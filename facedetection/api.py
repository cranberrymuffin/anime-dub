from __future__ import print_function
import numpy as np
import cv2
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

import pathlib


class FaceAlignment:
    def __init__(self):
        cascade_file = "lbpcascade_animeface.xml"
        cascade_file = pathlib.Path(__file__).parent.absolute() / cascade_file
        if not cascade_file.is_file():
            raise RuntimeError("%s: not found" % str(cascade_file))
        self.cascade = cv2.CascadeClassifier(str(cascade_file))

    def get_detections_for_frame(self, frame):
        frame = np.array(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(24, 24))
        (x, y, w, h) = faces[0]
        return frame[y:y+h, x:x+w]

    def get_detections_for_batch(self, frames):
        face_detections = []
        for frame in frames:
            face_detections.append(self.get_detections_for_frame(frame))
        return face_detections
