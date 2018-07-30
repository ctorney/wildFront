
import cv2
import numpy as np
import pandas as pd
import os, sys
import re
import math
from math import pi,sin,cos
import time



np.set_printoptions(precision=3,suppress=True)

DATAHOME = '/home/staff1/ctorney/data/wildebeest_front/'
inputname = DATAHOME + '/wildfront.csv'
dfMovies = pd.read_csv(inputname)

input_width = 4096
input_height = 2160


camera_matrix = np.array( [[  2467.726893, 0,  1936.02964], [0, 2473.06961, 1081.48243], [0, 0,1.0]])
dc = np.array( [ -1.53501973e-01,3.04457563e-01,8.83127622e-05,6.93998940e-04,-1.90560255e-01])

f = 12447
dist = 800


S = (4096,2160)

w = S[0]
h= S[1]
f=camera_matrix[0,0]
A1 = np.array([[1, 0, -w/2],[0,1, -h/2],[0, 0,0],[0, 0,1]])
#A1 = np.array([[f, 0, -camera_matrix[0,2]],[0,f, -camera_matrix[1,2]],[0, 0,0],[0, 0,1]])

       


        
A2 = np.array([[f, 0, w/2,0],[0, f, h/2, 0], [0,0,1,0]])

A2[0:3,0:3]=camera_matrix
      





warp_mode = cv2.MOTION_HOMOGRAPHY
number_of_iterations = 20

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-6;

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)


for index,  d in dfMovies.iterrows():

    # setup rotation matrix
    angle = float(d['angle'])
    print(angle)
    alpha = (90. - angle)*pi/180;
    R = np.array([[1,0,0,0],[0, cos(alpha), -sin(alpha), 0],[0,sin(alpha), cos(alpha),0],[0,0,0,1]])
    T = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0,0, 1, dist],[0, 0, 0, 1]])
    M2 = np.dot(A2,np.dot(T, np.dot(R,A1)))

    filename = DATAHOME + d['filename']
    cap = cv2.VideoCapture(filename)
    #noext, ext = os.path.splitext(filename)
    direct, ext = os.path.split(filename)
    noext, _ = os.path.splitext(ext)
    outfile = direct + '/proc/' + noext + '_stab.avi'
    output = direct + noext + '_WARP.npy'
    
    nx=4096
    ny=2160

# open the video
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print(fps)

    fStop = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    S = (4096,2160)

# reduce to 6 frames a second - change number to required frame rate
    ds = math.ceil(fps/6)

    out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('X','V','I','D'), fps/ds, (4096,2160), True)


      
    im1_gray = np.array([])
    first = np.array([])
    warp_matrix = np.eye(3, 3, dtype=np.float32) 
#warp_matrix = np.eye(2, 3, dtype=np.float32) 
    ts_full_warp = np.zeros((fStop//ds,3, 3), dtype=np.float32)
    i=0
    full_warp = np.eye(3, 3, dtype=np.float32)
    sys.stdout.write("\nProcessing movie " + filename + " \n=====================================\n" )
    sys.stdout.flush()
    for tt in range(fStop):
        start_all = time.time()
        # Capture frame-by-frame
        _, in_frame = cap.read()

        if (tt%ds!=0):
            continue
        frame = cv2.undistort(in_frame,camera_matrix,dc)
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%% %d " % ('='*int(20*tt/float(fStop)), int(100.0*tt/float(fStop)),tt))
        sys.stdout.flush()
        if not(im1_gray.size):
            # enhance contrast in the image
            im1_unwarp = cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
            im1_gray = cv2.warpPerspective(im1_unwarp, M2, (S[0],S[1]),flags=cv2.WARP_INVERSE_MAP)
            first = frame.copy()
        
        im2_unwarp =  cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        im2_unwarp[im2_unwarp>200]=0
        im2_gray = cv2.warpPerspective(im2_unwarp, M2, (S[0],S[1]),flags=cv2.WARP_INVERSE_MAP)
 #       cv2.imwrite(direct + '/proc/' + str(tt)+'stab.jpg',im2_gray)
        

        start = time.time()
        try:
            mask = np.zeros_like(im2_gray)
            mask[im2_gray>0]=1
            # find difference in movement between this frame and the last frame
            (cc, warp_matrix) = cv2.findTransformECC(im1_gray,im2_gray,warp_matrix, warp_mode, criteria, inputMask=mask)    
            # this frame becames the last frame for the next iteration
            im1_gray = im2_gray.copy()
        except cv2.error as e:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
            
        stop = time.time()
        print('ecc time: ', stop-start)
        
        # alll moves are accumalated into a matrix
        #full_warp = np.dot(full_warp, np.vstack((warp_matrix,[0,0,1])))
        #full_warp = np.dot(full_warp, warp_matrix)
        full_warp = np.dot(warp_matrix,full_warp)
        combined = np.linalg.inv(np.dot(M2,full_warp))
        ts_full_warp[i,:,:]=combined[:]
        i = i + 1
        # create an empty image like the first frame
        im2_aligned = np.zeros_like(frame)
#        np.copyto(im2_aligned, first)
        # apply the transform so the image is aligned with the first frame and output to movie file
        #im2_aligned = cv2.warpAffine(frame, full_warp[0:2,:], (S[0],S[1]), dst=im2_aligned, flags=cv2.INTER_LINEAR  , borderMode=cv2.BORDER_TRANSPARENT)
 #       im2_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]), dst=im2_aligned, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
        im2_aligned = cv2.warpPerspective(frame, combined, (S[0],S[1]), dst=im2_aligned, borderMode=cv2.BORDER_TRANSPARENT)
        out.write(im2_aligned)
        #cv2.imwrite(str(tt)+'stab.jpg',im2_aligned)
        stop_all = time.time()
        print('total time: ', stop_all-start_all)

    np.save(output,ts_full_warp)
    cap.release()
  #  out.release()



