import os
import os.path as osp
import sys
import imageio
import numpy as np
import math
import imgaug as ia
import matplotlib.pyplot as plt
import cv2
from imgaug import augmenters as iaa


def augmentation(images):
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # in 50% of all cases. In all other cases they will sample new values

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            #iaa.Fliplr(0.6), # horizontally flip 50% of all images
            #iaa.Flipud(0.2), # vertically flip 20% of all images

            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    #sometimes(iaa.Superpixels(p_replace=(0, 0.05), n_segments=(175, 200))), # convert images into their superpixel representation

                    iaa.GaussianBlur((0, 2.5)), # blur images with a sigma between 0 and 3.0
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    #iaa.SimplexNoiseAlpha(iaa.OneOf([
                    #    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    #    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                   # ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),

                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.9, 1.1), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-1, 1),
                            first=iaa.Multiply((0.9, 1.1), per_channel=True),
                            second=iaa.LinearContrast((0.8, 1.5))
                        )
                    ]),
                    iaa.LinearContrast((0.8, 1.5), per_channel=0.5), # improve or worsen the contrast
                    sometimes(iaa.ElasticTransformation(alpha=(0.15, 0.75), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.015))), # sometimes move parts of the image around
                    #sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    aug_imgs = seq.augment_images(images)
    return aug_imgs

def get_image(line):
    line = line.split()
    img = line[0]
    image = cv2.imread(img, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = imageio.imread(img)
    name = img.split("\\")[5]
    return image, name



if __name__ == '__main__':
    #''''
    images_path = 'train.txt'
    imgName_Path = 'D:\GitHub_Repository\Data\VOC2028Helmet\ImageSets\Main\\train.txt'
    output_dir = 'D:\GitHub_Repository\Data\VOC2028Helmet\\augmentation\\'
    with open(images_path) as f:
        lines = f.readlines()  # Lines 是读进来的train.txt文件中所有rows image地址及box信息数据
    # 分批次处理图片
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    Batchsize = 128
    n = len(lines)

    iter_num = math.ceil(n/Batchsize)
    for batch_num in range(iter_num):
        i = 0
        image_data = []
        image_name = []
        # 取图片进入缓存
        if batch_num != iter_num - 1:
            while(i != Batchsize):
                image, name = get_image(lines[Batchsize * batch_num + i])  # 送train.txt的某一行数据
                image_data.append(image)
                image_name.append(name)
                i = (i + 1)
        else: #最后一个iteration的图片数目小于batch_size，防止超出line的索引
            while(i != (n - Batchsize * (iter_num - 1))):
                image, name = get_image(lines[Batchsize * batch_num + i])  # 送train.txt的某一行数据
                image_data.append(image)
                image_name.append(name)
                i = (i + 1)


        aug_imgs = augmentation(image_data)

        j = 0
        len = np.size(aug_imgs)
        while j < len :
            output_img = aug_imgs[j]
            name = image_name[j]
            cv2.imwrite(output_dir + name , output_img)
            print("saving image:" + name)
            j += 1
    #'''''

    ''''
    # Load and display random sample and their bounding boxes
    img = "000104.jpg"
    image = cv2.imread(img,1)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    #ia.imshow(image)


    # 对每一张照片使用以下所有的变换，变换采用的scale从给的范围内随机选取
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25)),
        # iaa.AdditiveGaussianNoise(scale=(10, 60)),
        iaa.GaussianBlur((0, 3.0)),
        iaa.Crop(percent=(0, 0.2)),
        iaa.Fliplr(0.6),  # 0.5左右的图片翻转，
    ], random_order=True)

    # test on the same image as above
    # imggrid = augmentation.augment_images(image)
    images = [image, image, image, image]

    rotate = iaa.Affine(rotate=(-25, 25))
    flip = iaa.Fliplr(0.5),
    # ia.seed(3)
    # seq_det = seq.to_deterministic()
    # image_aug = rotate.augment_image(image)
    images_aug = seq.augment_images(images)

    ia.imshow(np.hstack(images_aug))
    # ia.imshow(image_aug)
    '''''