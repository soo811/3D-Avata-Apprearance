import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import sys
import dlib

def alignment(path):
    detector = dlib.get_frontal_face_detector()
    model = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    img = dlib.load_rgb_image(path)
    
    dets = detector(img, 1)
    
    objs = dlib.full_object_detections()
    
    for detection in dets:
        s = model(img, detection)
        objs.append(s)
        
    faces = dlib.get_face_chips(img, objs, size = 256, padding=1.0)
    
    return faces