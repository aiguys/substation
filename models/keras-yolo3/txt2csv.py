import os
import sys
import csv

annotations_path = 'D:\GitHub_Repository\mAP\mAP\input\ground-truth\\'
detections_Path = 'D:\GitHub_Repository\mAP\mAP\input\detection-results\\'
output_dir = 'D:\GitHub_Repository\mAP\Mean-Average-Precision-for-Boxes-master\Mean-Average-Precision-for-Boxes-master\\'

anno = open(output_dir + 'annotation.csv', 'w', newline='')
det = open(output_dir + 'detection.csv', 'w', newline='')

# used to get csv output file for performance evaluation
def text2csv(path, csv_filename):
    csv_writer = csv.writer(csv_filename)
    for image_id in os.listdir(path):
       with open(path + image_id,'r') as f:
           lines = f.readlines()  # Lines 是读进来的image.txt文件中所有的box信息数据
           n = len(lines)
           print("processing on", image_id)
           if not n:
               line = [image_id]
               csv_writer.writerow(line)
           else:
               for line in lines:
                   #line = line.split("聽")
                   #line[4] = line[4].split("\n")[0]

                   line = line.split(" ")
                   line[5] = line[5].split("\n")[0]
                   line.insert(0,image_id)
                   csv_writer.writerow(line)

       f.close()

if __name__ == '__main__':
    #text2csv(annotations_path,anno)
    text2csv(detections_Path,det)