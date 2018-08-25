import os, sys, glob, math, csv

import cv2
import numpy as np
import pandas as pd


np.set_printoptions(precision=3,suppress=True)

##########################################################################
##          movie files and parameters
##########################################################################
DATAHOME = '/home/staff1/ctorney/data/wildebeest_front/'
inputname = DATAHOME + '/wildfront.csv'
dfMovies = pd.read_csv(inputname)


width = 4096
height = 2160
output_file = 'out.avi'

camera_matrix = np.array( [[  2467.726893, 0,  1936.02964], [0, 2473.06961, 1081.48243], [0, 0,1.0]])
dc = np.array( [ -1.53501973e-01,3.04457563e-01,8.83127622e-05,6.93998940e-04,-1.90560255e-01])

dfMovies = dfMovies[dfMovies['ir']==0]

##########################################################################
##          loop through movies in list
##########################################################################
for index,  d in dfMovies.iterrows():


    if index!=11:
        continue
    filename = DATAHOME + d['filename']
    direct, ext = os.path.split(filename)
    noext, _ = os.path.splitext(ext)
    inputdatafile = direct + '/proc/' +  noext + '_POS.txt'
    warpsfile = direct + '/proc/' +  noext + '_WARP.npy'
    data = np.genfromtxt(inputdatafile,delimiter=',') #,dtype=None, names=True)
    timepoints=data[:,0]
    w_ids=(data[:,1])
    xpos = 0.5*(data[:,2]+data[:,4])
    ypos = 0.5*(data[:,3]+data[:,5])
    
    sys.stdout.write("\nProcessing movie " + filename + " \n=====================================\n" )
    sys.stdout.flush()





    results = []
    cap = cv2.VideoCapture(filename)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    S = (width,height)
    # reduce to 4 frames a second - change number to required frame rate
    ds = math.ceil(fps/4)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('X','V','I','D'), fps//ds, S, True)
    frame_idx=-1
    nframes = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    warps = np.load(warpsfile)

    for i in range(nframes):
        ret, in_frame = cap.read() 


        if (i%ds!=0):
            continue
        if i>(2*24*60):
            break
        frame_idx+=1
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*i/float(nframes)), int(100.0*i/float(nframes))))
        sys.stdout.flush()
        frame = cv2.undistort(in_frame,camera_matrix,dc)

        warp = warps[frame_idx]
        print('============')
        print(frame_idx)
        print('============')
        print(warp)
        print('============')
        #warp = np.eye(3, 3, dtype=np.float32) 
       # for det in detections:
       #     dt = det.to_tlbr()
        thisInds = timepoints==frame_idx
        thisID = w_ids[thisInds]
        thisXP = xpos[thisInds]
        thisYP = ypos[thisInds]
        im_aligned = np.zeros_like(frame)
 #       im_aligned = frame.copy()
        im_aligned = cv2.warpPerspective(frame, warp, (S[0],S[1]), dst=im_aligned, borderMode=cv2.BORDER_TRANSPARENT)

        for w in range(len(thisXP)):
 #           if thisID[w]!=1:
 #               continue
 #           cv2.circle(im_aligned, (int(thisXP[w]), int(thisYP[w])),5,(255,255,255), 2)
            cv2.putText(im_aligned, str(thisID[w]), (int(thisXP[w]), int(thisYP[w])),0, 5e-3 * 200, (0,255,0),2)

 #           iwarp = np.linalg.inv(warp)
 #           centre = np.expand_dims([thisXP[w],thisYP[w]], axis=0)
 #           centre = np.expand_dims(centre,axis=0)
 #           centre = cv2.perspectiveTransform(centre,iwarp)[0,0,:]
 #           cv2.circle(im_aligned, (int(centre[0]), int(centre[1])),5,(255,255,255), -1)

 #               cv2.putText(frame, str(track.track_id),(int(minx), int(miny)),0, 5e-3 * 200, (0,255,0),2)

        out.write(im_aligned)


    out.release()
    cap.release()

