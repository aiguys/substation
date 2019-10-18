# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 08:28:53 2018

@author: Peng Dezhi
"""

import xml.etree.ElementTree as ET
import os
import os.path as osp
import io
import sys
import matplotlib.pyplot as plt
import numpy as np

# This method is used to get label information from xml files. format '<class_name> <left> <top> <right> <bottom>'
# left:Xmin  right:Xmax  top:Ymin  Bottom: Ymax
def valid_annotation_label(filepath,filename):
    output_dir = 'D:\GitHub_Repository\Data\VOC2028Helmet\labels'
    if os.path.exists(filepath) == False:
        print(filepath+' :not found')
        return

    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    print("processing on file",filename)
    f = open(output_dir + "\\" + os.path.splitext(filename)[0] + ".txt", 'w')
    tree = ET.parse(filepath)
    AllCoodinate = []
    AllName = []
    # 获得根节点
    root = tree.getroot()
    for bbox in root.iter('bndbox'):
        coodinate = []
        for child in bbox:
            coodinate.append(int(child.text))
        AllCoodinate.append(coodinate)

    i = 0
    for obj in root.iter('object'):
        name = obj.find('name')
        xmin, ymin = AllCoodinate[i][0], AllCoodinate[i][1]
        xmax, ymax = AllCoodinate[i][2], AllCoodinate[i][3]
        string = name.text + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)
        i = i + 1
        f.write(string + '\n')
        #AllName.append(obj.find('name'))


        
    return

# This method is used get label class summary in numbers from xml files
def AnnotationEDA(filepath, hatBox, personBox):
    if os.path.exists(filepath) == False:
        print(filepath+' :not found')
    tree = ET.parse(filepath)
    objs = tree.findall('object')
    for obj in objs:
        if obj.find('name').text == 'hat':
            hatBox = hatBox + 1
        if obj.find('name').text == 'person':
            personBox = personBox + 1

    return hatBox, personBox

if __name__ == '__main__':
    wd = os.getcwd()
    ano_dir =wd + '\Annotations'
    filename_list = os.listdir(ano_dir)
    hatBox = 0
    personBox = 0
    for filename in filename_list:
        filepath = os.path.join(ano_dir, filename)
        valid_annotation_label(filepath,filename)
        hatBox, personBox = AnnotationEDA(filepath,hatBox,personBox)

    print("Hat:",hatBox)
    print("Person:",personBox)

    index = np.arange(len((hatBox,personBox)))
    bar_width = 0.3
    subjects = (u'hat', u'person')

    plt.xticks(index, subjects)
    plt.title(u'annotation_classes')
    plt.bar(index, (hatBox,personBox), bar_width, color='#0072BC', label='annotation_classes')
    plt.show()
            
            
