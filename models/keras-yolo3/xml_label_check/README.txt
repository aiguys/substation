put valid_label.py into data folder's path, run it to see the summary of different label 
classes in XML files. And it has function to generate lable(ground-truth) for mAP later,
and generating 'noAnnotationTest.txt' file for detection results for mAP also.
Also this file has some EDA work to see the distribution of data among differen anchors,
summary data classes distribution among pictures.

put voc_annotation.py into data folder's path, run it to get 'train.txt', 'text.txt' etc.
put DataPreparation.py into data folder's path, run it to get 'train.txt', 'text.txt' etc. 
'DataPreparation.py' It has function like extracting specific  classes from xml file(eg. all classes but 'person'); 
randomly select some background images written into train file path; 


SuffixCap2Low is used to solve some bug errors in images' name, 
Low image name suffix from .JPG to .jpg.