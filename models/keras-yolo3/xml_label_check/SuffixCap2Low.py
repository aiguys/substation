import os

''''
This script aims to solve the problem of files' suffix type, for some reason
some of images files have capital suffix type like '.JPG' which is unreadable
by model from 'train.py'. So this script will solve this problem.
'''''

#wd = 'D:\GitHub_Repository\Data\VOC2028Helmet'
path = 'D:\GitHub_Repository\Data\VOC2028Helmet\JPEGImages'

def Cap2LowSuffix():

    for img in os.listdir(path):
        img_id = os.path.splitext(img)[0]
        suffix = os.path.splitext(img)[1]
        #print(img_id)
        #print(suffix)

        if suffix == '.JPG':
            print(img_id + " suffix will be low soon")
            newname = img_id + ".jpg"

            os.chdir(path)
            os.rename(img, newname)

            print("All done!")


if __name__ == '__main__':
    Cap2LowSuffix()