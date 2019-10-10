from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import os.path as osp
from keras import backend as K
from keras.layers import Input
from tensorflow.python.framework import graph_util,graph_io

import sys
sys.path.append("..")
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from train import create_tiny_model,get_classes,get_anchors,create_model
#路径参数
#input_path = '/home/jtl/keras-yolo3-master/h5_to_pb/'
weight_file = 'ep007-loss33.746-val_loss28.975.h5'
weight_file_path = 'E:\project\substation\models\keras-yolo3\logs\\000\ep007-loss33.746-val_loss28.975.h5' # yolov3-tiny4hat.h5

output_graph_name = weight_file[:-3] + '.pb'

#模型config
classes_path = 'cfg/hat_classes.txt'  # 注意txt文档中的class_name顺序与voc_annotation.py sets中顺序一致
anchors_path = 'cfg/tiny_yolo_anchors.txt'  # 选择对应的yolo或tiny anchors
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)


#转换函数
def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = False):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir,model_name),output_dir)

def h5_to_tflite(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = False):

    # Convert the model. h5 to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)
    tflite_model = converter.convert()

    #graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the TensorFlow Lite model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    # Test the TensorFlow model on random input data.
    h5_results = h5_model(tf.constant(input_data))

    # Compare the result.
    for tf_result, tflite_result in zip(h5_results, tflite_results):
      np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)

if __name__ == '__main__':
    #输出路径
    output_dir = 'D:\GitHub_Repository\substation\models\keras-yolo3\model_data'#osp.join(os.getcwd(),"trans_model")
    #加载模型
    image_input = Input(shape=(None, None, 3))
    num_anchors = len(anchors)
    h5_model = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    # load_weights用于加载keras框架外用户自定义的模型框架的权重
    h5_model.load_weights(weight_file_path)
    # load_model用于加载keras框架内定义的模型框架的权重
    #h5_model = load_model(weight_file_path)
    ## h5_to_pb(h5_model,output_dir = output_dir,model_name = output_graph_name)
    h5_to_tflite(h5_model,output_dir = output_dir,model_name = output_graph_name)
    ##print('model saved')


