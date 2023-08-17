import urllib
import cv2
import dlib
import math
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import tensorflow as tf
# from tensorflow import keras
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.preprocessing.image import ImageDataGenerator


## 헤어 흑백 분할
def hair_seg(path):
    IMAGE_FILENAMES = []
    IMAGE_FILENAMES.append(path)
    
    BG_COLOR = (0,0,0)
    MASK_COLOR = (255, 255, 255)
    
    base_options = python.BaseOptions(model_asset_path='hair_segmenter.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                          output_category_mask = True)
    
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        
        for image_file_name in IMAGE_FILENAMES:
            
            image = mp.Image.create_from_file(image_file_name)
            
            segmentation_result = segmenter.segment(image)
            category_mask = segmentation_result.category_mask
            
            # image_data = image.numpy_view()
            fg_image = np.zeros((3,), dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros((3,),dtype=np.uint8)
            bg_image = BG_COLOR
            
            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition, fg_image, bg_image)
            
    return output_image


## 데이터 증강
def imgGen(images, labels, size):
    tf.random.set_seed(42)
    image_generator = ImageDataGenerator(
        rotation_range=15,
        brightness_range=[0.8, 1.0],
        zoom_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=False
    )

    augment_size = size

    np.random.seed(42)

    random_mask = np.random.randint(images.shape[0], size=augment_size)
    image_aug = images[random_mask].copy()
    label_aug = labels[random_mask].copy()

    image_aug = image_generator.flow(image_aug, np.zeros(augment_size), batch_size=augment_size, shuffle=False, seed=42).next()[0]

    images = np.concatenate((images, image_aug))
    labels = np.concatenate((labels, label_aug))

    return images, labels


## test 이미지 전처리(contour & padding & resize)
def testimg(size, img):
    image = np.array(img, dtype=np.uint8)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    contour, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = np.concatenate(contour)

    x,y,w,h = cv2.boundingRect(cnts)
    
    crop_img = image[y:y+h, x:x+w]

    BG_COLOR = (0, 0, 0)

    DEFAULT = size

    if crop_img.shape[0] > crop_img.shape[1]:
        DEFAULT = crop_img.shape[0]
    else:
        DEFAULT = crop_img.shape[1]

    height, width, channel = crop_img.shape

    w_x = int((DEFAULT - width) / 2)
    h_y = int((DEFAULT - height) / 2)

    img_pad = cv2.copyMakeBorder(crop_img, h_y, h_y, w_x, w_x, cv2.BORDER_CONSTANT, value=[0,0,0])

    w_x2 = DEFAULT - img_pad.shape[1]
    h_y2 = DEFAULT - img_pad.shape[0]

    img_pad = cv2.copyMakeBorder(img_pad, h_y2, 0, w_x2, 0, cv2.BORDER_CONSTANT, value=[0,0,0])

    img_pad = cv2.resize(img_pad, (size, size))

    ## 바로 prediction에 넣을 수 있도록
    test_image = []
    test_image.append(img_pad)
    test_image = np.array(test_image) / 255.0

    return test_image



## 눈썹 이미지 전처리
def eyebrow(path):
    face_origin = cv2.imread(path)
    face_copy = face_origin.copy()
    face_rgb = cv2.cvtColor(face_copy, cv2.COLOR_BGR2RGB)

    detector = dlib.get_frontal_face_detector()
    dlib_rects = detector(face_rgb, 1)

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    img = []

    for dlib_rect in dlib_rects:
        points = predictor(face_rgb, dlib_rect)
        list_points = list(map(lambda p: [p.x, p.y], points.parts()))
        img.append(list_points)
        img = np.array(img[0])

    xs = img[22:27,0]
    ys = img[22:27,1]

    x = min(xs)
    y = min(ys)
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)

    crop_img = face_copy[y:y+h, x:x+w]
    bw = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    invert_img = cv2.bitwise_not(bw)
    binary_image = cv2.adaptiveThreshold(invert_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 49, 2)
    
    DEFAULT = 64
    if binary_image.shape[0] > binary_image.shape[1]:
            DEFAULT = binary_image.shape[0]
    else:
        DEFAULT = binary_image.shape[1]

    BG_COLOR = (0, 0, 0)

    height, width = binary_image.shape

    w_x = int((DEFAULT - width) / 2)
    h_y = int((DEFAULT - height) / 2)

    img_pad = cv2.copyMakeBorder(binary_image, h_y, h_y, w_x, w_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    w_x2 = DEFAULT - img_pad.shape[1]
    h_y2 = DEFAULT - img_pad.shape[0]

    test_img = cv2.copyMakeBorder(img_pad, h_y2, 0, w_x2, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    test_img= cv2.resize(test_img, (64, 64))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    ## 바로 prediction에 넣을 수 있도록
    test_image = []
    test_image.append(test_img)
    test_image = np.array(test_image) / 255.0

    return test_image
