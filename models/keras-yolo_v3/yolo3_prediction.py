# -*- coding: utf-8 -*-
# @Time    : 2019-09-16 11:25
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : yolo3_prediction.py
# @Description:

"""Demo for use yolo v3
"""
import os
import time
import cv2
import numpy as np
from yolo import YOLO, detect_video


def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()


def detect_image(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image


def detect_vedio(video, yolo, all_classes):
    """Use yolo v3 to detect video.

    # Argument:
        video: video file.
        yolo: YOLO, yolo model.
        all_classes: all classes name.
    """
    camera = cv2.VideoCapture(video)
    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)

    while True:
        res, frame = camera.read()

        if not res:
            break

        image = detect_image(frame, yolo, all_classes)
        cv2.imshow("detection", image)

        if cv2.waitKey(110) & 0xff == 27:
                break

    camera.release()


if __name__ == '__main__':

    file = '/Users/henry/Documents/application/keras-yolo3/model_data/coco_classes.txt'
    all_classes = get_classes(file)

    parameters = {
        "model_path": '/Users/henry/Documents/application/keras-yolo3/model_data/yolo3.h5',
        "anchors_path": '/Users/henry/Documents/application/keras-yolo3/model_data/yolo_anchors.txt',
        "classes_path": '/Users/henry/Documents/application/keras-yolo3/model_data/coco_classes.txt',
        #"score": 0.3,
        #"iou": 0.45,
        #"model_image_size": (416, 416),
        #"gpu_num": 1,
    }
    yolo = YOLO(**parameters)
    # detect images in test floder.
    for (root, dirs, files) in os.walk('/Users/henry/Desktop'):
        if files:
            for f in files:
                print(f)
                path = '/Users/henry/Desktop/1.jpg'#os.path.join(root, f)
                image = cv2.imread(path)
                image = detect_image(image, yolo, all_classes)
                cv2.imwrite('images/res/' + f, image)

    # detect vedio.
    video = 'E:/video/car.flv'
    detect_vedio(video, yolo, all_classes)