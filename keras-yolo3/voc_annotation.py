import xml.etree.ElementTree as ET
from os import getcwd

#sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
sets=['train','val','test'] # generate 'train','val','test' txt files
classes=["person", "hat"] # the custom classes in images，person with '0',hat with '1'

def convert_annotation(image_id, list_file):
  #  in_file = open('D:\GitHub_Repository\Data\VOC2028Helmet\Annotations\%s.xml'%(image_id),encoding="utf-8") #打开对应image_id的xml文件(尽量别用中文地址目录，否则要加,encoding="utf-8")
    in_file = open('/home/student/project/xj/data/VOC2028Helmet/Annotations/%s.xml'%(image_id),encoding="utf-8")
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes:  # or difficult = 1 pass the small difficult hats
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

#wd = getcwd()
#wd = 'D:\GitHub_Repository\Data\VOC2028Helmet'
wd = '/home/student/project/xj/data/VOC2028Helmet'

for  image_set in sets:
   # if not os.path.exists('./labels/'):
    #    os.makedirs('./labels/')
    image_ids = open('/home/student/project/xj/data/VOC2028Helmet/ImageSets/Main/%s.txt'%(image_set)).read().strip().split() # 读取Main下的txt文件中的所有images id
    list_file = open('%s.txt'%( image_set), 'w') #以写模式创建txt文件，train.txt,test.txt
    for image_id in image_ids:
      #  list_file.write('%s\JPEGImages\%s.jpg'%(wd, image_id)) # 写入image地址
        list_file.write('%s/JPEGImages/%s.jpg'%(wd, image_id)) # 写入image地址
        convert_annotation(image_id,list_file)
        list_file.write('\n')
    list_file.close()

