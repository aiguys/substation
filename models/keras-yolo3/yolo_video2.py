import sys
import argparse
from yolo import YOLO
from PIL import Image
import base64
from io import BytesIO
import video_detect
FLAGS = None
#import setproctitle

def detect_img(yolo):
    while True:
        img = '/Users/henry/Desktop/1.jpg'#input('/Users/henry/Desktop/1.jpg')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image,out_boxes, out_scores, out_classes = yolo.predict(image,{67})#yolo.detect_image(image)
            r_image.show()
    yolo.close_session()
    return r_image,out_boxes, out_scores, out_classes





if __name__ == '__main__':

    #setproctitle.setproctitle('yolo3')
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file '
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions '
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions '
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use '
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='/Users/henry/Desktop/test2/1.mp4',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="/Users/henry/Desktop/test/",
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
            yolo.detect_image(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        #detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
        yolo = YOLO(**vars(FLAGS))
        video_detect.detect_camera3(yolo,videoPath=FLAGS.input,output_path=FLAGS.output, loop=5,trackingAlt=False,isDebug=True)
        yolo.close_session()

        #detect_video_concurrent(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
