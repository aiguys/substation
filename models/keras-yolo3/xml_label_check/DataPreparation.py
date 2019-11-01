import os
import numpy as np
import xml.etree.ElementTree as ET

## This py file is used to generate train.txt and test.txt files
'''' # rename all image files with its specific image_id
for root, dirs, files in os.walk(wd):
    count = 0
    for file in files:
        print(file)
        Old_path = os.path.join(root,file)
        file = file[-18:]
        path = os.path.join(root, file)
        os.rename(Old_path, path)
        count += 1
    print("一共修改了" + str(count) + "个文件")
'''''

def txtFilePreparation():
    wd = os.getcwd()
    pic_path = wd + '\JPEGImages'
    anno_path = wd + '\Annotations'
   # pic_path = wd + '\JPEGImages'
    JPEGImage_txt = 'JPEGImages.txt'
    pic_path_name = '\JPEGImages'

    # classes = ["hat", "nohat", "glove", "noglove", "boots", "noboots", "safetybelt", "nosafetybelt", "person"]
    # classes = ["hat", "nohat", "glove", "noglove", "boots", "noboots", "safetybelt", "nosafetybelt"]
    classes = ["hat","nohat","safetybelt", "nosafetybelt"]
    # classes = ['hat','person']
    #classes = ['hat', 'nohat']

    list_file = open(JPEGImage_txt, 'w')
    background_file = open('background.txt', 'w')
    train_files = open('ImageSets\Main_belt&Hat\\train.txt', 'w')
    test_files = open('ImageSets\Main_belt&Hat\\test.txt', 'w')


    #''' comment when deal with Main_moreImages
    # get 5% background random txt files written into 'JPEGImages.txt'
    background_path = wd + '\\background'
    files = os.listdir(background_path)
    for image_id in files:
        background_file.write('%s\\background\%s\n' % (wd, image_id))

    background_split = 0.2
    path = os.path.join(wd,'background.txt')
    with open(path) as f:
        lines = f.readlines()  # Lines background.txt文件中所有rows
    np.random.seed(24)
    np.random.shuffle(lines)
    num_background = int(len(lines) * background_split)  # 20% as val from shuffled background.txt(150 imgs)

    for backgroundImg_path in lines[:num_background]:
        list_file.write(backgroundImg_path)
    #'''

    # ***************get train and test txt files******************
    files = os.listdir(pic_path)
    for image_id in files:
        list_file.write('%s' % (wd) + pic_path_name + '\%s\n' % (image_id))

    test_split = 0.1
    path = os.path.join(wd,JPEGImage_txt)
    with open(path) as f:
        lines = f.readlines()  # Lines JPEGImages.txt文件中所有rows image地址
    np.random.seed(24)
    np.random.shuffle(lines)
    num_test = int(len(lines) * test_split)  # 10% as val from shuffled train.txt
    num_train = len(lines) - num_test

    for trainImg_path in lines[:num_train]:
        trainImg_path = trainImg_path.strip('\n')
        if trainImg_path.split("\\")[-2] == 'background':
            train_files.write(trainImg_path)
            train_files.write('\n')
            continue

        image_id = trainImg_path
        image_id = image_id.split("\\")[-1]
        time_ = ' '
        if len(image_id.split('-')) > 4: # select data with time_prefix '20191024'
            time_ = image_id.split('-')[2]
        image_id = image_id[:-4]  # 去除.jpg

        if os.path.exists(anno_path +'\%s.xml' % (image_id)):

            if ifAllUnwanttedclass(image_id, classes, anno_path):
                # do nothing but skip this file
                continue
            train_files.write(trainImg_path)


            convert_annotation(image_id, train_files, classes, anno_path, time_)
            train_files.write('\n')

    for testImg_path in lines[num_train:]:
        testImg_path = testImg_path.strip('\n')
        if testImg_path.split("\\")[-2] == 'background':
            test_files.write(testImg_path)
            test_files.write('\n')
            continue

        image_id = testImg_path
        image_id = image_id.split("\\")[-1]
        image_id = image_id[:-4]  # 去除.jpg

        if os.path.exists(anno_path + '\%s.xml' % (image_id)):
            testImg_path = testImg_path.strip("\n")
            if ifAllUnwanttedclass(image_id, classes, anno_path):
                # do nothing but skip this file
                continue
            test_files.write(testImg_path)
            convert_annotation(image_id, test_files, classes, anno_path, time_) # write down annotations behind image file_path
            test_files.write('\n')


def convert_annotation(image_id, list_file, classes, anno_path, time_):
    in_file = open(anno_path + '\%s.xml'%(image_id),'r',encoding='UTF-8') # %(year, image_id)
    print("processing on image: ", image_id)


    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        # we don't want more data for 'hat' and 'nohat' from new data now
        if cls in ['hat','nohat'] and time_ == '20191024':
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id)) #str(1)所有的person写id ’1‘


def ifAllUnwanttedclass(image_id, classes, anno_path):
    in_file = open(anno_path + '\%s.xml' % (image_id), 'r', encoding='UTF-8')  # %(year, image_id)

    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in classes:
            # if find any one cls is included in classes set
            return False

    return True


if __name__ == '__main__':
    txtFilePreparation()

    ''' # shuffle train.txt
    path = 'D:\GitHub_Repository\Data\substation\\train.txt'
    list_file = open('shuffled_train.txt', 'w')
    with open(path) as f:
        lines = f.readlines()  # Lines JPEGImages.txt文件中所有rows image地址
    np.random.seed(24)
    np.random.shuffle(lines)
    for line in lines:
        list_file.write(line)
    '''





