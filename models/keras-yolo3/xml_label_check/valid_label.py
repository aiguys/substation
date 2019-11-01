# -*- coding: utf-8 -*-
"""
Created on Fri Nov 1 14:28:53 2019

@author: Jun Xiao
"""

## This py file is used to generate noAnnotationtest.txt file and label folder(ground-truth)
## Also, it has method for EDA work, summary of classes' number
import xml.etree.ElementTree as ET
import os
import os.path as osp
import io
import sys
import matplotlib.pyplot as plt
import numpy as np


# This method is used to get label information from xml files. format '<class_name> <left> <top> <right> <bottom>'
# left:Xmin  right:Xmax  top:Ymin  Bottom: Ymax
def valid_annotation_label(output_dir, classes, filepath, filename, bool_background=False, bool_trick = False):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)

    if bool_background:  # background image has no xml file, we just save its name without annotations
        print("processing on background file", filename)
        f = open(output_dir + "\\" + filename + ".txt", 'w')
        # f.write(filename + '\n')

    else:

        print("processing on file", filename)
        os.path.splitext(filename)[0]
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
            cls = obj.find('name').text
            if cls not in classes:
                # if find any cls is not included in classes set
                continue
            xmin, ymin = AllCoodinate[i][0], AllCoodinate[i][1]
            xmax, ymax = AllCoodinate[i][2], AllCoodinate[i][3]
            if bool_trick and cls == 'person': # when class is ['hat', 'person'] we regard 'person' as 'nohat' for VOC dataset
                string = 'nohat' + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)
            else:
                string = cls + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)

            i = i + 1
            f.write(string + '\n')
            # AllName.append(obj.find('name'))
    return


# This method is used get label class summary in numbers from xml files
# box_list: personBox, hatBox, nohatBox, gloveBox, nogloveBox, bootsBox, nobootsBox, safetybeltBox, nosafetybeltBox
def AnnotationEDA(filepath, box_list, trickFlag= False):
    if os.path.exists(filepath) == False:
        print(filepath + ' :not found')
    tree = ET.parse(filepath)
    objs = tree.findall('object')
    for obj in objs:
        if obj.find('name').text == 'person':
            if not trickFlag:
                box_list[0] = box_list[0] + 1
            else:
                box_list[2] = box_list[2] + 1 # in voc dataset 'person' cls should be counted as nohat
        elif obj.find('name').text == 'hat':
            box_list[1] = box_list[1] + 1
        elif obj.find('name').text == 'nohat':
            box_list[2] = box_list[2] + 1
        elif obj.find('name').text == 'glove':
            box_list[3] = box_list[3] + 1
        elif obj.find('name').text == 'noglove':
            box_list[4] = box_list[4] + 1
        elif obj.find('name').text == 'boots':
            box_list[5] = box_list[5] + 1
        elif obj.find('name').text == 'noboots':
            box_list[6] = box_list[6] + 1
        elif obj.find('name').text == 'safetybelt':
            box_list[7] = box_list[7] + 1
        elif obj.find('name').text == 'nosafetybelt':
            box_list[8] = box_list[8] + 1

    return box_list

def EDA_helper(testfiles_list, anoAndTestfiles_list):

    classes = ['person', 'hat', 'nohat', 'glove', 'noglove', 'boots', 'noboots', 'safetybelt', 'nosafetybelt']

    # below it is for EDA work, using annotation xml file to count number for all class
    # make sure you define the classes correctly
    map_pair = []
    map_pair.append(classes)
    map_pair.append([0] * len(classes))

    # open every annotation direction and open all test files inside
    for i in range(len(testfiles_list)):
        dum_list = testfiles_list[i]
        if i == 1:  # when voc data set
            trickFlag = True
        else:
            trickFlag = False
        for filename in dum_list:
            filepath = os.path.join(anoAndTestfiles_list[0][i], filename)
            print("processing on: " + filepath)
            # valid_annotation_label(filepath,filename)
            map_pair[1] = AnnotationEDA(filepath, map_pair[1], trickFlag)

    # the index array of our classes
    # require change every time
    data = (
    map_pair[1][0], map_pair[1][1], map_pair[1][2], map_pair[1][3], map_pair[1][4], map_pair[1][5], map_pair[1][6],
    map_pair[1][7], map_pair[1][8])
    index = np.arange(
        len((map_pair[1])))
    bar_width = 0.5
    subjects = (classes)

    # plt.xticks(index, subjects)
    x_label = 'number of object per class'
    plt.xlabel(x_label, fontsize='large')
    plt.title(u'annotation_classes')
    # colors = ['red', 'yellow'] #(map_pair[1][0], map_pair[1][1]
    plt.barh(index, data,
             bar_width, color='skyblue', tick_label=classes, label='annotation_classes')
    # plt.bar(index,
    #        (personBox, hatBox, nohatBox, gloveBox, nogloveBox, bootsBox, nobootsBox, safetybeltBox, nosafetybeltBox),
    #        bar_width, color='#0072BC', label='annotation_classes')
    plt.show()



def get_labelsAndTestnoAnnotationtxt(Main_path, output_dir):
    # select the claimed classes written into label files
    classes = ['hat', 'nohat', 'glove', 'noglove', 'boots', 'noboots', 'safetybelt', 'nosafetybelt', 'person']
    # classes = ['hat', 'nohat']

    Testfilename_list = Main_path + '\\test.txt'

    # test_file is test file without annotation behind file path
    test_file = open(Main_path + "\\test_noAnnotation.txt", 'w')
    with open(Testfilename_list) as f:
        lines = f.readlines()  # Lines 是读进来的text.txt文件中所有rows image地址及box信息数据
    for text_id in lines:
        text_id = text_id.strip('\n')
        sub_dir = text_id.split('\\')[-2]
        text_id = text_id.split('\\')[-1]
        text_id = text_id.split(' ')[0]
        text_id = text_id[:-4]

        if sub_dir == 'background':  # background file has no xml file respectively
            test_file.write(sub_dir + '\\' + text_id + '\n')  # write background image path into test_file
            valid_annotation_label(output_dir, classes, filepath, text_id,
                                   bool_background=True)  # create background test annotation file
            continue
        elif sub_dir == 'JPEGImages':
            text_filename = text_id + '.xml'
            filepath = os.path.join(ano_dir, text_filename)
            valid_annotation_label(output_dir, classes, filepath, text_filename)
            # This is used to get test file with only image_id written into noAnnotation.txt
            test_file.write(text_id + '\n')
        elif sub_dir == 'moreJPEGImages':
            text_filename = text_id + '.xml'
            filepath = os.path.join(moreAnno_dir, text_filename)
            valid_annotation_label(output_dir, ['hat', 'person'], filepath, text_filename, bool_background=False,
                                   bool_trick=True)
            # This is used to get test file with 'moreJPEGImages\\'+ image_id written into noAnnotation.txt
            test_file.write(sub_dir + '\\' + text_id + '\n')



if __name__ == '__main__':
    wd = os.getcwd()
    ano_dirList = []
    testfiles_list = []
    anoAndTestfiles_list = []
    # below two path is for output train and test txt files in method'get_labelsAndTestnoAnnotationtxt'
    output_dir = 'D:\GitHub_Repository\Data\substation\ImageSets\Main_onlyhat\labels'
    Main_path = 'D:\GitHub_Repository\Data\substation\ImageSets\Main_onlyhat'

    ano_dir = wd + '\Annotations'
    ano_dirList.append(ano_dir)
    moreAnno_dir = wd + '\moreAnnotations'
    ano_dirList.append(moreAnno_dir)
    anoAndTestfiles_list.append(ano_dirList)

    filename_list = os.listdir(ano_dir)
    testfiles_list.append(filename_list)
    dum_list = os.listdir(moreAnno_dir)
    testfiles_list.append(dum_list)
    anoAndTestfiles_list.append(testfiles_list)

    #EDA_helper(testfiles_list, anoAndTestfiles_list)
    get_labelsAndTestnoAnnotationtxt(Main_path, output_dir)


