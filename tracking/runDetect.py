

import cv2
import numpy as np
import time
from deepModels import getSegModel
from matplotlib import pyplot as plt
import pandas as pd
import os

np.set_printoptions(precision=3,suppress=True)

DATAHOME = '/home/ctorney/data/tz-2017/'
CODEHOME = '/home/ctorney/workspace/wildFront/'
inputname = CODEHOME + '/irMovieList.csv'
dfMovies = pd.read_csv(inputname,index_col=0)







for index,  d in dfMovies.iterrows():

    filename = d['filename']
    print(filename)
    w_size = d['w_size']

    rescale = 40.0/float(w_size)
    print(rescale)
    bZ = 4
    tnx = int((rescale*4096)/bZ)
    tny = int((rescale*2160)/bZ)
    print(tny,tnx) 
    modelSeg = getSegModel(tnx,tny)

    filepath='training/class_weights.hdf5'
    modelSeg.load_weights(filepath)
    cap = cv2.VideoCapture(filename)
    noext, ext = os.path.splitext(filename)
    outfile = noext + '_ML.avi'
    out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), (4096,2160), True)
    
    frCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    nx=4096
    ny=2160
#    output = 0*255.0*np.ones((ny,nx),dtype=np.float32)
#    border = 32
#    #set border to zero
#    output[0:border,:]=0
#    output[ny-border:ny,:]=0
#    output[:,0:border]=0
#    output[:,nx-border:nx]=0
#    
    for tt in range(frCount):
        print(tt)
        _, frame = cap.read()
        if frame is None:
            break
        
        cframe = cv2.resize(frame,(bZ*tnx,bZ*tny))# aNone,fx=rescale,fy=rescale)
 #       input_height,input_width,_ = cframe.shape//2
        input_height = tny
        input_width = tnx

        im = cframe.astype('float32')/255.0
        output = np.zeros((bZ*tny,bZ*tnx))

        for i in range(bZ):
            for j in range(bZ):
                predictions=modelSeg.predict(np.reshape(im[i*input_height:(i+1)*input_height,j*input_width:(j+1)*input_width,:],(1,input_height,input_width,3)))[0]


#bb=model.predict(np.reshape(X_test[:,:42,],(1,42,42,3)))
#print(bb)

#aaa=np.reshape(aa[:,:,1],(111,157))
 #       output=255*np.reshape(np.argmax(predictions,2),(input_height//4,input_width//4)).astype(np.float32)
                print(predictions.shape)
                output[i*input_height:(i+1)*input_height,j*input_width:(j+1)*input_width] = np.argmax(predictions, axis=2)
 #       output = cv2.resize(output,(2*input_width,2*input_height), cv2.INTER_LINEAR)
        output = cv2.resize(output,(nx,ny), cv2.INTER_LINEAR)
        outRGB = cv2.cvtColor(255*output.astype(np.uint8),cv2.COLOR_GRAY2BGR)
        cv2.imwrite('test.png',outRGB)
  #      output = cv2.resize(output,(input_width,input_height), cv2.INTER_LINEAR)
  #      outRGB = cv2.cvtColor(output.astype(np.uint8),cv2.COLOR_GRAY2BGR)
        out.write(outRGB)
        break

                

    
            
            
    
    
    
    cap.release()
    out.release()
    
    break

