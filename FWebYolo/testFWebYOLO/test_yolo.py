# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/16 17:05
# Name:         test_yolo
# Description:

import requests
import os
import json
if __name__ == '__main__':
    url = 'http://192.168.0.214:9000/yolo_service'
    headers = {'Content-Type': 'image/jpeg'}

    file = {'media': open('1.jpg', 'rb')}
    requests.post(url, files=file)
    
    data = open('1.jpg', 'rb').read()
    r = requests.post(url, data=data, headers=headers, verify=False)
    print("类别：置信度 (左上角X坐标,左上角Y坐标) (目标的宽, 目标的高)")
    print(r.text)