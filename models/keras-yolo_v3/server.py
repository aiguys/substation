# -*- coding: utf-8 -*-
# @Time    : 2019-09-17 11:45
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : server.py
# @Description:用于flask服务

# -*- coding: utf-8 -*-
# @Time    : 2019-08-16 13:02
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : extractor.py
# @Description:
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash,jsonify
import os
import json
import setproctitle
CSRF_ENABLED = True
from yolo import YOLO, detect_video
from PIL import Image
import base64
from io import BytesIO
FLAGS = None

app = Flask(__name__, instance_path='/Users/henry/Documents/application/keras-yolo3/instance/folder')
app.config.from_object('config')
print('app.config=', app.config)
yolo = YOLO(** {key.lower(): value for key, value in app.config.items() if key in {'MODEL_PATH','ANCHORS_PATH','CLASSES_PATH'}})
print('yolo3 initialized')

def detect(image,input=None,output=None):
    r_image = None
    out_boxes = None
    out_scores = None
    out_classes = None
    if image :
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if input is not None :
            print(" Ignoring remaining command line arguments: " + input + "," + output)

        r_image, out_boxes, out_scores, out_classes = yolo.predict2(image)  # yolo.detect_image(image)


    elif "input" in FLAGS:

         pass
    else:
        print("Must specify at least video_input_path.  See usage with --help.")

    return  r_image, out_boxes, out_scores, out_classes


@app.route('/detection', methods=['GET', 'POST'])
def detection():
    """

    :param base64Image: base64转码后的图片
    :return:
    """
    base64Image = None

    if request.method == 'POST':
        print('POST ')
        rj = request.get_json()
        base64Image = rj['base64Image']
    elif request.method == 'GET':
        print('GET')
        base64Image = request.args['base64Image']

    #print('base64Image=',base64Image)

    byte_date = base64.b64decode(base64Image)
    try:
        imagede = Image.open(BytesIO(byte_date))
    except Exception as e:
        print('Open Error! Try again!')
        raise e
    r_image, out_boxes, out_scores, out_classes = detect(imagede)
    #r_image.show()
    imagede.close()
    retData = []
    print("len(out_boxes) = {}, len(out_scores)={},len(out_classes)={}".format(len(out_boxes),len(out_scores),len(out_classes)))
    for (b,s,c) in zip(out_boxes, out_scores, out_classes):
        retData.append({'out_box':b.tolist(),'out_score':float(s) ,'out_class':int(c)})
    return json.dumps(retData)





@app.route('/detectvedio/')
def detectvedio():
    #input,output
    return  detect(image=None,input=None,output=None)


@app.route('/', methods=['GET', 'POST'])
def show_template(name=None):
    base64Image = None
    if request.method == 'POST':
        print('POST ')
        rj = request.get_json()
        base64Image = rj['base64Image']
    else:
        print('GET')
        base64Image = request.args['base64Image']

    print('base64Image=',base64Image)
    inputText=base64Image
    return render_template('404.html',inputContent=inputText)



if __name__ == '__main__':
    setproctitle.setproctitle('yolo3')
    #detection()
    # class YOLO defines the default value, so suppress any default here

    #app.run(
    #    host='0.0.0.0',
    #    port=9990,
        #debug=True#Flask配置文件在开发环境中，在生产线上的代码是绝对不允许使用debug模式，正确的做法应该写在配置文件中，这样我们只需要更改配置文件即可但是你每次修改代码后都要手动重启它。这样并不够优雅，而且 Flask 可以做到更好。如果你启用了调试支持，服务器会在代码修改后自动重新载入，并在发生错误时提供一个相当有用的调试器。
   # )
    app.run(debug=app.config['DEBUG'],host=app.config['HOST'], port=app.config['PORT'])
    print('------------------------run end')



