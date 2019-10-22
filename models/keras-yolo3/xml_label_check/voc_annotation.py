import xml.etree.ElementTree as ET
from os import getcwd

# sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets = ['train','test','val']

classes = ["person", "hat"]


def convert_annotation(image_id, list_file):
    in_file = open('./Annotations/%s.xml'%(image_id),encoding='UTF-8') # %(year, image_id)
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for image_set in sets:
    image_ids = open('./ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s.txt'%(image_set), 'w')
    if image_set=='train':
        aug_file = open('%s.txt' % (image_set + '_Aug'), 'w')
    for image_id in image_ids:
        list_file.write('%s\JPEGImages\%s.jpg'%(wd, image_id))
        print("processing on image: ", image_id)
        convert_annotation(image_id, list_file)
        if image_set=='train':
            aug_file.write('%s\\augmentation\%s.jpg' % (wd, image_id))
            convert_annotation(image_id, aug_file)
            aug_file.write('\n')
        list_file.write('\n')
    list_file.close()

