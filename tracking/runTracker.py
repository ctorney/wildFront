import os, sys, glob, math, csv

import cv2
import numpy as np
import pandas as pd
from deep_sort import nn_matching
from deep_sort.detection import Detection
from yolo_detector import yoloDetector
from deep_sort.tracker import Tracker



##########################################################################
##          movie files and parameters
##########################################################################
DATAHOME = '/home/staff1/ctorney/data/wildebeest_front/'
inputname = DATAHOME + '/wildfront.csv'
dfMovies = pd.read_csv(inputname)


width = 4096
height = 2160
max_cosine_distance = 0.5
outputVideo = True
output_file = 'out.avi'

camera_matrix = np.array( [[  2467.726893, 0,  1936.02964], [0, 2473.06961, 1081.48243], [0, 0,1.0]])
dc = np.array( [ -1.53501973e-01,3.04457563e-01,8.83127622e-05,6.93998940e-04,-1.90560255e-01])

dfMovies = dfMovies[dfMovies['ir']==0]

##########################################################################
##          loop through movies in list
##########################################################################
for index,  d in dfMovies.iterrows():


    filename = DATAHOME + d['filename']
    direct, ext = os.path.split(filename)
    noext, _ = os.path.splitext(ext)
    outputdatafile = direct + '/proc/' +  noext + '_POS.txt'
    warpsfile = direct + '/proc/' +  noext + '_WARP.npy'
    
    sys.stdout.write("\nProcessing movie " + filename + " \n=====================================\n" )
    sys.stdout.flush()


    ######################################################################
    ##          set-up yolo detector and tracker
    ######################################################################
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance)
    tracker = Tracker(metric,max_age=12)#,max_iou_distance=1.0)
    yolo = yoloDetector(width,height, 'weights/balloon-yolo.h5')



    results = []
    cap = cv2.VideoCapture(filename)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    S = (width,height)
    # reduce to 4 frames a second - change number to required frame rate
    ds = math.ceil(fps/4)
    if outputVideo:                     
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('X','V','I','D'), fps//ds, S, True)
    frame_idx=0
    nframes = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    warps = np.load(warpsfile)

    for i in range(nframes):
        ret, in_frame = cap.read() 

        if (i%(fps*60*1)==0):
            tracker = Tracker(metric,max_age=12)#,max_iou_distance=1.0)

        if (i%ds!=0):
            continue
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*i/float(nframes)), int(100.0*i/float(nframes))))
        sys.stdout.flush()
        frame = cv2.undistort(in_frame,camera_matrix,dc)

        warp = warps[frame_idx]
        #warp = np.eye(3, 3, dtype=np.float32) 
        detections = yolo.create_detections(frame, warp)
        # Update tracker.
        tracker.predict()
        tracker.update(detections)
       # for det in detections:
       #     dt = det.to_tlbr()

        for track in tracker.tracks:
            bbox = track.to_tlbr()
            if not track.is_confirmed():
                continue 
            if track.time_since_update >0 :
                continue

            if outputVideo:
                iwarp = np.linalg.inv(warp)
                bwidth = bbox[2]-bbox[0]
                bheight = bbox[3]-bbox[1]
                centre = np.expand_dims([0.5*(bbox[0]+bbox[2]),0.5*(bbox[1]+bbox[3])], axis=0)
                centre = np.expand_dims(centre,axis=0)
                centre = cv2.perspectiveTransform(centre,iwarp)[0,0,:]
  #              if not track.is_confirmed():
  #                  print('uc:', bbox[0],bbox[1],bbox[2],bbox[3], corner1[0],corner1[1])
  #              else:
  #                  print('c:', bbox[0],bbox[1],bbox[2],bbox[3], corner1[0],corner1[1])
             #   corner2 = np.expand_dims([[bbox[2],bbox[3]]], axis=0)
             #   corner2 = cv2.perspectiveTransform(corner2,iwarp)[0,0,:]
                minx = centre[0]-bwidth*0.5
                maxx = centre[0]+bwidth*0.5
                miny = centre[1]-bheight*0.5
                maxy = centre[1]+bheight*0.5

                cv2.putText(frame, str(track.track_id),(int(minx), int(miny)),0, 5e-3 * 200, (0,255,0),2)
   #             if bbox[1]>0:
                cv2.rectangle(frame, (int(minx), int(miny)), (int(maxx), int(maxy)),(255,0,0), 4)
    #            else:
    #                cv2.rectangle(frame, (int(corner1[0]), int(corner1[1])), (int(corner2[0]), int(corner2[1])),(0,0,255), 4)

            results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
        frame_idx+=1

        if outputVideo:
            out.write(frame)


    if outputVideo:
        out.release()
    cap.release()

    with open(outputdatafile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(results)
