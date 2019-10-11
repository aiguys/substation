import base64
import random
from io import BytesIO
import sys
import numpy as np
import colorsys
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
import os

Classification_CLASS_PATH = 'E:\GitHub_Repository\substation\models\keras-yolo3\cfg\hat_classes.txt' #cf.get("class_graph_path")
Detection_PATH_TO_FROZEN_GRAPH = 'E:\GitHub_Repository\substation\models\keras-yolo3\model_data\ep036-loss21.932-val_loss16.520.pb'# '#cf.get("graph_path")



"""

def center_crop(x, center_crop_size):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped = x[centerw - halfw: centerw + halfw,
              centerh - halfh: centerh + halfh, :]

 


def pad_to_bounding_box(image, target_height, target_width):
    # is_batch = True
    image_shape = image.shape

    height, width, depth = (image_shape[0], image_shape[1], image_shape[2])
    after_padding_width_left = target_width // 2 - width // 2 + (target_width - width) % 2
    after_padding_width_right = target_width // 2 - width // 2
    after_padding_height_top = target_height // 2 - height // 2 + (target_height - height) % 2
    after_padding_height_bottom = target_height // 2 - height // 2

    # Do not pad on the depth dimensions.
    paddings = np.reshape(
        np.stack([after_padding_height_top, after_padding_height_bottom, after_padding_width_left,
                  after_padding_width_right, 0, 0]), [3, 2])

    padded = np.pad(image, paddings, 'constant')

    return padded

 
# 读取图片,并得到固定大小
def scale_byRatio(img_path, return_width=200, crop_method=center_crop, defor=None):
    # img_base64 = base64.b64decode(img_path)
    # nparr = np.fromstring(img_base64, np.uint8)
    img_crop_np = np.asarray(img_path)
    h, w, _ = img_crop_np.shape

    img_resized = cv2.resize(img_crop_np, (0, 0), fx=return_width / w, fy=return_width / h,
                             interpolation=cv2.INTER_CUBIC)
    img_rgb = img_resized

    return img_rgb
"""

def load_image_into_numpy_array(img):
    nparr = np.fromstring(img, np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # h, w, _ = img.shape
    # img_resized = cv2.resize(img, (return_width, return_width), interpolation=cv2.INTER_CUBIC)
    # p = np.array(img, dtype=np.float32)
    (w, h) = img.size
    p = np.array(img.getdata()).reshape((h, w, 3)).astype(np.uint8)
    p = np.expand_dims(p, axis=0)
    return p


def load_batch_my(image, height=100, width=100, is_training=False):
    # image_raw = cv2.imread(image_path)
    image_raw_tensor = tf.convert_to_tensor(image)
    # Preprocess image for usage by Inception.
    if (image_raw_tensor.dtype != tf.float32):
        image_raw_tensor = tf.image.convert_image_dtype(image_raw_tensor, dtype=tf.float32)
    if height and width:
        # Resize the image to the specified height and width.
        image_raw_tensor = tf.expand_dims(image_raw_tensor, 0)
        image_raw_tensor = tf.image.resize_bilinear(image_raw_tensor, [height, width], align_corners=False)
        image_raw_tensor = tf.squeeze(image_raw_tensor, [0])
    image_raw_tensor = tf.subtract(image_raw_tensor, 0.5)
    image_raw_tensor = tf.multiply(image_raw_tensor, 2.0)
    # Preprocess the image for display purposes.

    with tf.device('/gpu:0'):
        with tf.Session() as sess:
            image_raw_tensor = image_raw_tensor.eval()
        image_raw_tensor = np.expand_dims(image_raw_tensor, 0)

    return image_raw_tensor, image


def run_inference_for_detection(image_path, sess):
    """
    目标检测
    # tf.Graph对象定义了一个命名空间对于它自身包含的tf.Operation对象
# TensorFlow自动选择一个独一无二的名字，对于数据流图中的每一个操作
# 但是给操作添加一个描述性的名字可以使你的程序更容易来阅读和调试
# TensorFlow api提供两种方式来重写一个操作的名字
# 1、每一个api函数创建了一个新的tf.Operation或者返回一个新的tf.Tensor接收一个可选的name参数
#     列如tf.constant(42.0,name="answer")创建了一个新的tf.Operation叫做answer
#     并且返回以叫做"answer:0"的tf.Tensor .如果默认的数据流图已经包含了叫做"answer"的操作
#     TensorFlow会在后面append,"_1","_2"来使它变得唯一
# 2、tf.name_scope函数使得在名字后面添加一个后缀变得可能
    :param image_path:
    :param sess:
    :return:
    """
    ops = sess.graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = sess.graph.get_tensor_by_name(
                tensor_name)
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')

    # Run inference
    image_np_expanded = load_image_into_numpy_array(image_path)
    #tf.Session.run()函数返回值为fetches的执行结果。如果fetches是一个元素就返回一个值；若fetches是一个list，则返回list的值，若fetches是一个字典类型，则返回和fetches同keys的字典。
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    return output_dict




def get_class(classes_path):
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

def getSession():
    try:
        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.visible_device_list = '0'
        config.gpu_options.per_process_gpu_memory_fraction = 0.1

        with tf.gfile.FastGFile(Detection_PATH_TO_FROZEN_GRAPH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='kellytest')
            sess1 = tf.Session(config=config)

        return sess1
    except:
        print("Unexpected error:", sys.exc_info())

def getclass_namesandcolos():
    class_names = get_class(Classification_CLASS_PATH)
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                      colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    return class_names,colors

def go(base64Image):
    sess1 = getSession()

    img_base64 = base64.b64decode(base64Image)
    image_data = BytesIO(img_base64)
    image = Image.open(image_data)
    output_dict = run_inference_for_detection(image, sess1)
    out_boxes = output_dict['detection_boxes']
    out_classes = output_dict['detection_classes']
    out_scores = output_dict['detection_scores']
    class_names, colors = getclass_namesandcolos()

    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300



    for i, c in reversed(list(enumerate(out_classes))):

        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw



    return (image,out_boxes, out_scores, out_classes)


def detect_img(base64Image):
    # result = vehicle_result(base64Image)
    result = go(base64Image)

    return result
if __name__ == '__main__':
    base64Image = 'person.jpg'
    detect_img(base64Image)