#import xml.etree.ElementTree as ET  
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir,getcwd
from os.path import join
'''
output: a lot of label files in labels folder and 'train,val and test' .txt files. 
Each file for one image jpg file. Format as <object-class> <x> <y> <width> <height>, 
if the image has object-class the first position will be filled with '1' otherwise '0'.
'''

#sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]  

#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]  

sets=['train','val','test'] # generate 'train','val','test' txt files
#sets = ["test"]
classes=["person", "hat"] # the custom classes in images


def convert(size,box):
    dw=1./size[0]
    dh=1./size[1]
    x=(box[0]+box[1])/2.0
    y=(box[2]+box[3])/2.0
    w=box[1]-box[0]
    h=box[3]-box[2]
    x=x*dw
    w=w*dw
    y=y*dh
    h=h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('./Annotations/%s.xml'%( image_id),encoding="utf-8")
    out_file = open('./ImageSets/labels_name/%s.txt'%( image_id),'w',encoding="utf-8")
    tree=ET.parse(in_file)
    root=tree.getroot()
    size=root.find('size')
    w=int(size.find('width').text)
    h=int(size.find('height').text)

    for obj in root.iter('object'):
        difficult=obj.find('difficult').text
        cls=obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id=classes.index(cls)       
        xmlbox=obj.find('bndbox')
        b=(float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),float(xmlbox.find('ymin').text),float(xmlbox.find('ymax').text))
        bb=convert((w,h),b)
        #out_file.write(str(cls_id)+" "+" ".join([str(a) for a in bb]) +'\n')
        if cls_id == 1:
            out_file.write("hat"+" "+" ".join([str(a) for a in bb]) +'\n')
        else :
            out_file.write("person"+" "+" ".join([str(a) for a in bb]) +'\n')
        
        

# Get the Root address
wd=getcwd() # 得到当前目录
#wd = '/home/jxiao/object-detection/Data/VOC2028Helmet'

for  image_set in sets:
    if not os.path.exists('./ImageSets/labels_name/'):
        os.makedirs('./ImageSets/labels_name/')
    image_ids=open('./ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file=open('%s.txt'%(image_set),'w',encoding="utf-8") # 创建train.txt，val.txt,test.txt
    for image_id in image_ids:
        list_file.write('%s\\augmentation\%s.jpg\n' % (wd, image_id))  # 写入txt文件每个图片的地址
        # list_file.write('%s\JPEGImages\%s.jpg\n'%(wd, image_id)) # 写入txt文件每个图片的地址
        #list_file.write('%s/JPEGImages/%s.jpg\n'%(wd, image_id)) # 写入txt文件如 ‘/home/jxiao/object-detection/Data/VOC2028Helmet/JPEGImages/000999.jpg’
        convert_annotation(image_id)
    list_file.close()

