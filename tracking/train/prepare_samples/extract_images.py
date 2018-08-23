import cv2
import glob
import numpy as np
import os

# set the number of training samples to create
n_sample = 100

# set the directory to save the images, this will save to a
# directory called train_ims that should be in the current directory
sample_dir = '../train_images_1/'

# create a list of all movies to use to create the samples, this will make
# a list of all files that end with MP4 in the current directory - should
# be changed to the file format used (avi, mpg etc)

filelist = glob.glob("/home/ctorney/data/tz-2017/vis/DJI_0363.MP4")

# the following lines determine how many images to take from each movie
# first determine how many movies in total
count = len(filelist) 
# next integer divide to get the number of images per movie rounded down to nearest integer
fpm = n_sample//count 
# make a list of how many frames per movie to save
counts = fpm*np.ones(count,dtype=int)
# increase the count by one for some movies so we get exactly 1000
for i in range(n_sample-fpm*count):
    counts[i]+=1

# this code loops over each movie and using opencv extracts 
# the designated number of frames chosen at random
for i in range(count):
    # get the movie filename
    m_name = filelist[i]
    print(m_name)
    # get the number of frames to save
    m_count= counts[i]
    # get the moviename without the extension to use as the image filename
    head, tail = os.path.split(m_name)
    noext, ext = os.path.splitext(tail)
    
    # open the movie and get its frame count
    cap = cv2.VideoCapture(m_name)
    m_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # choose the frames to select at random
    f_list =np.sort(np.random.choice(m_length, m_count, replace=False))
    
    # go through the frame list, read the frame and then save to a png file   
    for f_num in f_list:
        print(f_num)
        cap.set(cv2.CAP_PROP_POS_FRAMES,f_num)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(sample_dir + noext +'-'+ str(f_num) + '.png',frame)
            print(sample_dir + noext +'-'+ str(f_num) + '.png')
    
