from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import os.path as osp
from keras import backend as K
from keras.layers import Input
from tensorflow.python.framework import graph_util,graph_io
from keras.backend import clear_session

import sys
sys.path.append("..")
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from train import create_tiny_model,get_classes,get_anchors,create_model
#路径参数
#input_path = '/home/jtl/keras-yolo3-master/h5_to_pb/'
weight_file = 'ep036val_loss16.520.h5'
weight_file_path = 'D:\GitHub_Repository\keras-yolo3\\logs\\000\ep036-loss21.932-val_loss16.520.h5' # yolov3-tiny4hat.h5

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



if __name__ == '__main__':
    #输出路径
    output_dir = 'D:\GitHub_Repository\substation\models\keras-yolo3\model_data'#osp.join(os.getcwd(),"trans_model")

    #加载模型(h5 to pb)
    image_input = Input(shape=(None, None, 3))
    num_anchors = len(anchors)
    h5_model = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    # load_weights用于加载keras框架外用户自定义的模型框架的权重
    h5_model.load_weights(weight_file_path)
    # load_model用于加载keras框架内定义的模型框架的权重
    ## h5_model = load_model(weight_file_path)
    h5_to_pb(h5_model,output_dir = output_dir,model_name = output_graph_name)
    print('model saved')

    ''' # 查看keras模型的输入、输出tensor名字
    clear_session()
    np.set_printoptions(suppress=True)
    image_input = Input(shape=(None, None, 3))
    num_anchors = len(anchors)
    input_graph_name = "model_data\ep036-loss21.932-val_loss16.520.h5"
    output_graph_name = weight_file_path[:-3] + '.tflite'
    h5_model = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    h5_model.load_weights(input_graph_name)
    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_file=h5_model, Custom=True)
    converter.post_training_quantize = True
    # 在windows平台这个函数有问题，无法正常使用
    tflite_model = converter.convert()
    open(output_graph_name, "wb").write(tflite_model)
    print ("generate:", output_graph_name)
    '''




