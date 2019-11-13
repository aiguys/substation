# -*- coding: utf-8 -*-
"""
Created on Fri Nov 1 14:28:53 2019

@author: Jun Xiao
"""

## This py file is used to generate noAnnotationtest.txt file and label folder(ground-truth)
## Also, it has method for EDA work, summary of classes' number, summary of data distribution among different anchors
import xml.etree.ElementTree as ET
import os
import os.path as osp
import io
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
                i += 1
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

    # 不要更改classes的设定，目前只支持所有9个类一起统计的方法
    classes = ['person', 'hat', 'nohat', 'glove', 'noglove', 'boots', 'noboots', 'safetybelt', 'nosafetybelt']
    #classes = ['person', 'hat', 'nohat', 'safetybelt', 'nosafetybelt']

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
    data = (map_pair[1][0], map_pair[1][1], map_pair[1][2], map_pair[1][3], map_pair[1][4],map_pair[1][5], map_pair[1][6], map_pair[1][7], map_pair[1][8])
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

# This method is used to summary data distribution among different anchors
def summary_clsDistribution_anchors():
    wd = 'D:\GitHub_Repository\Data\VOC2012'
    anno_name = 'personLayoutNewAnnotations'
    anno_path = wd + '\\' + anno_name + '\\'

    anchors = [12,14,  21,24,  31,35,  44,50,  62,71,  86,100,  117,155,  176,240,  309, 434]
    squre = []
    data = np.zeros([9,4])
    index = ['(%s,%s)'%(str(anchors[0]),str(anchors[1])), '(%s,%s)'%(str(anchors[2]),str(anchors[3])), '(%s,%s)'%(str(anchors[4]),str(anchors[5])),
             '(%s,%s)'%(str(anchors[6]),str(anchors[7])), '(%s,%s)'%(str(anchors[8]),str(anchors[9])), '(%s,%s)'%(str(anchors[10]),str(anchors[11])),
             '(%s,%s)'%(str(anchors[12]),str(anchors[13])), '(%s,%s)'%(str(anchors[14]),str(anchors[15])), '(%s,%s)'%(str(anchors[16]),str(anchors[17]))]
    columns = ['person','nohat','noglove','noboots']
    df = pd.DataFrame(data, index=index, columns = columns )
    #print(df.head())

    # 9 组anchors下的形成的矩形面积
    for i in range(int(len(anchors)/2)):
        s = anchors[i*2] * anchors[i*2 +1]
        squre.append(s)


    files_list = os.listdir(anno_path)
    for file in files_list:
        in_file = open(anno_path + file, 'r', encoding='UTF-8')  # %(year, image_id)
        tree = ET.parse(in_file)

        for obj in tree.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            s = (xmax - xmin) * (ymax - ymin)
            cls = obj.find('name').text

            if cls not in columns:
                continue

            # 找到任何满足条件的范围，在dataframe中这个cls的对应位置加1
            # break后查看下一个cls的num_cls
            for j in range(int(len(anchors)/2) - 2):
                if s <= (squre[0] + squre[1])/2:
                    df.at['(%s,%s)'%(str(anchors[0]),str(anchors[1])), cls] += 1
                    break
                elif s > (squre[j] + squre[j + 1])/2 and s <= (squre[j + 1] + squre[j + 2])/2:
                    df.at['(%s,%s)'%(str(anchors[j*2 + 2]),str(anchors[j*2 + 3])), cls] += 1
                    break
                elif s > (squre[7] + squre[8])/2:
                    df.at['(%s,%s)'%(str(anchors[16]),str(anchors[17])),cls] += 1
                    break

            ''''
            if s <= (squre[0] + squre[1])/2:
                df.at['(%s,%s)'%(str(anchors[0]),str(anchors[1])), cls] += 1
            elif s > (squre[0] + squre[1])/2 and s <= (squre[1] + squre[2])/2:
                df.at['(%s,%s)'%(str(anchors[2]),str(anchors[3])), cls] += 1
            elif s > (squre[1] + squre[2])/2 and s <= (squre[2] + squre[3])/2:
                df.at['(%s,%s)'%(str(anchors[4]),str(anchors[5])), cls] += 1
            elif s > (squre[2] + squre[3])/2 and s <= (squre[3] + squre[4])/2:
                df.at['(%s,%s)'%(str(anchors[6]),str(anchors[7])), cls] += 1
            elif s > (squre[3] + squre[4])/2 and s <= (squre[4] + squre[5])/2:
                df.at['(%s,%s)'%(str(anchors[8]),str(anchors[9])), cls] += 1
            elif s > (squre[4] + squre[5])/2 and s <= (squre[5] + squre[6])/2:
                df.at['(%s,%s)'%(str(anchors[10]),str(anchors[11])), cls] += 1
            elif s > (squre[5] + squre[6])/2 and s <= (squre[6] + squre[7])/2:
                df.at['(%s,%s)'%(str(anchors[12]),str(anchors[13])), cls] += 1
            elif s > (squre[6] + squre[7])/2 and s <= (squre[7] + squre[8])/2:
                df.at['(%s,%s)'%(str(anchors[14]),str(anchors[15])), cls] += 1
            else:
                df.at['(%s,%s)'%(str(anchors[16]),str(anchors[17])),cls] += 1
            '''

    df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1)
    df.loc['Row_sum'] = df.apply(lambda x: x.sum())
    print(df)
    #print(df.describe())
    df.to_csv(wd + "\summaryAnchors_{}.csv".format(anno_name), index_label="index_label")

# This method is used to summary data classes distribution among all images
# 'Col_sum' is sum number of all images that contains the specified class
# output example:
# 	      below 3	(3,5)	(5,10)	(10,15)	(15,20)	(20,25)	(25,30)	over 30	Col_sum
# person	564	     28	     14	      2	      1	       0	   0	 0	     609
def summary_clsDistribution_images():
    wd = 'D:\GitHub_Repository\Data\substation'
    anno_name = 'Annotations'
    anno_path = wd + '\\' + anno_name + '\\'

    cls_range = [3, 5, 10, 15, 20, 25, 30]
    #class_name = ['person','nohat','noglove','noboots']
    class_name = ['person','hat','nohat','glove','noglove','boots','noboots','safetybelt','nosafetybelt']
    data = np.zeros([len(class_name),len(cls_range)+1])
    columns = ['below %s'%(str(cls_range[0])), '(%s,%s)'%(str(cls_range[0]),str(cls_range[1])), '(%s,%s)'%(str(cls_range[1]),str(cls_range[2])),
             '(%s,%s)'%(str(cls_range[2]),str(cls_range[3])), '(%s,%s)'%(str(cls_range[3]),str(cls_range[4])), '(%s,%s)'%(str(cls_range[4]),str(cls_range[5])),
             '(%s,%s)'%(str(cls_range[5]),str(cls_range[6])), 'over %s'%(str(cls_range[6]))]
    df = pd.DataFrame(data, index=class_name, columns = columns)
    #print(df)
    class_sum = []
    class_sum.append(class_name)
    class_sum.append([0] * len(class_name))

    files_list = os.listdir(anno_path)
    for file in files_list:
        in_file = open(anno_path + file, 'r', encoding='UTF-8')  # %(year, image_id)
        tree = ET.parse(in_file)
        class_sum[1] = [0] * len(class_name) # 用于计数统计每一个file中所有obj的class

        for obj in tree.findall('object'):
            cls = obj.find('name').text
            for i in range(len(class_sum[0])):
                # if satisfied,switch to next object
                if cls == class_name[i]:
                    class_sum[1][i] += 1
                    break

        for i in range(len(class_sum[0])):
            num_cls = class_sum[1][i]
            if num_cls == 0:
                continue
            # 找到任何满足条件的范围，在dataframe中这个cls的对应位置加1
            # break后查看下一个cls的num_cls
            for j in range(len(cls_range) + 1):
                if num_cls <= cls_range[0]:
                    df.at[class_name[i], 'below %s' % (str(cls_range[0]))] += 1
                    break
                elif num_cls > cls_range[j] and num_cls <= cls_range[j + 1]:
                    df.at[class_name[i], '(%s,%s)' % (str(cls_range[j]), str(cls_range[j + 1]))] += 1
                    break
                elif num_cls > cls_range[len(cls_range) - 1]:
                    df.at[class_name[i], 'over %s' % (str(cls_range[len(cls_range) - 1]))] += 1
                    break

    df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1)
    #df.loc['Row_sum'] = df.apply(lambda x: x.sum())
    print(df)
    #print(df.describe())
    df.to_csv(wd + '\\' + "summary_{}.csv".format(anno_name), index_label="index_label")

# This method is used to summary and count image source
def summary_data_source():
    wd = 'D:\GitHub_Repository\Data\VOC2012'
    anno_path = wd + '\personLayoutNewAnnotations\\'
    # add new image or video source prefix title here
    DataSource = ['00','partB','part2','video-ownhat-20191024','mark-video','VOC2012']
    data = np.zeros([1, len(DataSource)])
    sum_list = [0] * len(DataSource)
    index = ['sum']
    df = pd.DataFrame(data, index=index, columns=DataSource)
    files_list = os.listdir(anno_path)

    for file in files_list:
        image_id = file[:-4]
        # '001368' 'video-ownhat-20191024'  'part2_002309' '2007_000032'

        prefix = image_id[:-7]
        prefix2 = image_id[:-4]
        prefix3 = image_id[:-9]
        if prefix == 'video-ownhat-20191024':
            sum_list[3] += 1
        elif prefix == 'PartB':
            sum_list[1] += 1
        elif prefix == 'part2':
            sum_list[2] += 1
        elif prefix2 == '00' and len(image_id) == 6:
            sum_list[0] += 1
        elif prefix3 == '20':
            sum_list[5] += 1
        else:
            sum_list[4] += 1

    df.loc['sum'] = sum_list
    print(df)



def get_labelsAndTestnoAnnotationtxt(Main_path, output_dir, classes):
    Testfilename_list = Main_path + '\\test.txt'
    filepath = ' '

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
            # 'moreJPEGImages'下的原始voc数据只有'hat', 'person'两类，后续修改标注的partB具有更多类别
            valid_annotation_label(output_dir, ['hat', 'person','safetybelt', 'nosafetybelt'], filepath, text_filename, bool_background=False,
                                   bool_trick=True)
            # This is used to get test file with 'moreJPEGImages\\'+ image_id written into noAnnotation.txt
            test_file.write(sub_dir + '\\' + text_id + '\n')



if __name__ == '__main__':

    #summary_clsDistribution_images()
    #summary_clsDistribution_anchors()
    summary_data_source()

    '''' Summary classes number or get test.txt and test_noAnnotation.txt files
    wd = os.getcwd()
    ano_dirList = []
    testfiles_list = []
    anoAndTestfiles_list = []
    # select the claimed classes written into label files
    classes = ['hat', 'nohat', 'glove', 'noglove', 'boots', 'noboots', 'safetybelt', 'nosafetybelt', 'person']
    #classes = ['hat', 'nohat', 'safetybelt', 'nosafetybelt','person']
    # below two path is for output train and test txt files in method'get_labelsAndTestnoAnnotationtxt'
    output_dir = 'D:\GitHub_Repository\Data\substation\ImageSets\Main_HBP\labels'
    Main_path = 'D:\GitHub_Repository\Data\substation\ImageSets\Main_HBP'

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

    EDA_helper(testfiles_list, anoAndTestfiles_list) # 统计‘Annotations’目录下的我们想看的所有labels
    #get_labelsAndTestnoAnnotationtxt(Main_path, output_dir, classes)
    '''''


