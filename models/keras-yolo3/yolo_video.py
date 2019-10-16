import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import time
import os

def detect_img(yolo):
    ''''
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        # 如果不抛出except的异常，执行else
        else:
            start = time.clock()
            r_image = yolo.detect_image(image)
            elapsed = (time.clock() - start)
            print("Time used:", elapsed)
            r_image.show()
    '''''

    wd = 'D:\GitHub_Repository\Data\VOC2028Helmet'

    if not os.path.exists(wd + '\ImageSets\detection-results'):
        os.makedirs(wd + '\ImageSets\detection-results')
    image_ids = open(
        'D:\GitHub_Repository\Data\VOC2028Helmet\ImageSets\Main\%s.txt' % ('test')).read().strip().split()
    for image_id in image_ids:
        try:
            #print(wd + '\JPEGImages\\' + image_id)
            image = Image.open(wd + '\JPEGImages\\%s.jpg'  %(image_id))
        except:
            print('Open Error!' + image_id)
            break
        else:
            list_file = open(wd + '\ImageSets\detection-results\%s.txt' % (image_id), 'w')  # 以写模式为每一张image创建txt文件
            start = time.clock()
            print("Inference and processing on " + image_id)
            yolo.detect_image(image,list_file)
            elapsed = (time.clock() - start)
            print("Time used:", elapsed)
    #'''''
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")

