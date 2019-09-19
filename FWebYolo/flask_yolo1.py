# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/16 14:12
# Name:         upload_pictures
# Description:

from flask import Flask, render_template, request, jsonify, make_response
from werkzeug.utils import secure_filename
import os
import cv2
import time
from PIL import Image
from io import BytesIO
import json
import numpy as np
import base64

import setproctitle

from datetime import timedelta
import yolomain

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


FWebYolo = {
    'test'
}
@app.route('/api/<func>', methods=['GET', 'POST'])
def go(func):
    exec('import fe.' + func)
    data = eval('fe.' + func + '.go()')
    return data


# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        # 当前文件所在路径
        basepath = os.path.dirname(__file__)

        # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        f.save(upload_path)

        lab, img, loc = yolomain.yolo_detect(pathIn=upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)

        return render_template('upload_ok.html', userinput=user_input, val1=time.time(), data_dict=lab)

    return render_template('upload.html')

@app.route('/yolo_service',methods=['POST', 'GET'])
def detection():
    base64Image = None

    if request.method == 'POST':
        print('POST ')
        rj = request.get_json()
        base64Image = rj['base64Image']
    elif request.method == 'GET':
        print('GET')
        base64Image = request.args['base64Image']

    # print('base64Image=',base64Image)

    byte_date = base64.b64decode(base64Image)
    try:
        imagede = Image.open(BytesIO(byte_date))
    except Exception as e:
        print('Open Error! Try again!')
        raise e

    image = np.array(imagede, dtype='float32')
    out_boxes, out_scores, out_classes = yolomain.yolo_ser(image)

    # rsp = make_response(json.dumps(lab))
    # rsp.mimetype = 'application/json'
    # rsp.headers['Connection'] = 'close'
    # return rsp

    imagede.close()
    retData = []
    print("len(out_boxes) = {}, len(out_scores)={},len(out_classes)={}".format(len(out_boxes), len(out_scores),
                                                                               len(out_classes)))
    for (b, s, c) in zip(out_boxes, out_scores, out_classes):
        retData.append({'out_box': b, 'out_score': float(s), 'out_class': c})
    return json.dumps(retData)

if __name__ == '__main__':
    setproctitle.setproctitle('FWebYolo')
    app.run(host='0.0.0.0', port=9000)
