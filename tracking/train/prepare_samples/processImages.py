import numpy as np
import pandas as pd
import os,sys,glob
import cv2
import pickle
sys.path.append("../..") 
from models.yolo_models import get_yolo_model
from decoder import decode
image_dir =  '../train_images_1/'
train_dir = '../train_images_1/'


train_images =  glob.glob( image_dir + "D*.png" )

width=4096
height=2160

im_size=864 #size of training imageas for yolo

nx = width//im_size
ny = height//im_size

##################################################
im_size=416 #size of training imageas for yolo
yolov3 = get_yolo_model(im_size,im_size,trainable=False)
yolov3.load_weights('../../weights/yolo-v3-coco.h5',by_name=True)
print(yolov3.summary())
im = cv2.imread('balloon.png')
new_image = im[:,:,::-1]/255.
new_image = np.expand_dims(new_image, 0)
aa = np.tile(new_image, (2,1,1,1))

            # run the prediction
yolos = yolov3.predict(aa)

boxes = decode(yolos, im_size, 'predictions.png', im)
sys.exit('bye!')



########################################
im_num=1
all_imgs = []
for imagename in train_images: 
    im = cv2.imread(imagename)
    print('processing image ' + imagename + ', ' + str(im_num) + ' of ' + str(len(train_images))  + '...')
    im_num+=1

    n_count=0
    for x in np.arange(0,width-im_size,im_size):
        for y in np.arange(0,height-im_size,im_size):
            img_data = {'object':[]}     #dictionary? key-value pair to store image data
            head, tail = os.path.split(imagename)
            noext, ext = os.path.splitext(tail)
            save_name = train_dir + '/TR_' + noext + '-' + str(n_count) + '.png'
            box_name = train_dir + '/bbox/' + noext + '-' + str(n_count) + '.png'
            img = im[y:y+im_size,x:x+im_size,:]
            cv2.imwrite(save_name, img)
            img_data['filename'] = save_name
            img_data['width'] = im_size
            img_data['height'] = im_size
            n_count+=1
            # use the yolov3 model to predict 80 classes on COCO

            # preprocess the image
            image_h, image_w, _ = img.shape
            new_image = img[:,:,::-1]/255.
            new_image = np.expand_dims(new_image, 0)

            # run the prediction
            yolos = yolov3.predict(new_image)

            boxes = decode(yolos, im_size, box_name, img)
            for b in boxes:
                xmin=b[0]
                xmax=b[2]
                ymin=b[1]
                ymax=b[3]
                obj = {}

                obj['name'] = 'aoi'

 #               xmin = point['xcoord'] - x - sz_2
 #               xmax = point['xcoord'] - x + sz_2
 #               ymin = point['ycoord'] - y - sz_2
 #               ymax = point['ycoord'] - y + sz_2

                if xmin<0: continue
                if ymin<0: continue
                if xmax>im_size: continue
                if ymax>im_size: continue
                obj['xmin'] = int(xmin)
                obj['ymin'] = int(ymin)
                obj['xmax'] = int(xmax)
                obj['ymax'] = int(ymax)
                img_data['object'] += [obj]

            all_imgs += [img_data]


#print(all_imgs)
with open(train_dir + '/annotations.pickle', 'wb') as handle:
   pickle.dump(all_imgs, handle)
                        

