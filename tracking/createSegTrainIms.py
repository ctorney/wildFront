import numpy as np
import sys,argparse
import os
import pandas as pd
import cv2
from deepModels import getModel
import scipy.ndimage as nd
from skimage import draw

DATAHOME = '/home/ctorney/data/tz-2017/'
CODEHOME = '/home/ctorney/workspace/wildFront/'
inputname = CODEHOME + '/irMovieList.csv'
dfMovies = pd.read_csv(inputname,index_col=0)

model = getModel()

segpath='training/segmentation/'
filepath='training/class_weights.hdf5'
model.load_weights(filepath)


imSize = 40

for index,  d in dfMovies.iterrows():

    filename = d['filename']
    w_size = d['w_size']
    
    
    rescale = 40.0/float(w_size)
    cap = cv2.VideoCapture(filename)
    noext, ext = os.path.splitext(filename)
    frCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for tt in range(frCount):

        _, frame = cap.read()
        if frame is None:
            break
        if (tt%600)!=0:
            continue

        outputname = segpath + 'MOVIE-' + str(index) + '-' + str(tt) 
        cframe = cv2.resize(frame,None,fx=rescale,fy=rescale)
        ny,nx,_ = cframe.shape
 #   nx=4096
 #   ny=2160
        output = np.zeros((ny,nx),dtype=np.float32)
        border = 2*int(w_size)
        
        batchSize = 4096*64
        stepSize = 2
        xpos = np.zeros(batchSize,dtype='int')
        ypos = np.zeros(batchSize,dtype='int')
        batch = np.zeros((batchSize,imSize,imSize,3),dtype=np.float32)
        b=0
        for y in range(border, cframe.shape[0]-border, stepSize):
            print(y)
            for x in range(border, cframe.shape[1]-border, stepSize):
                xpos[b]=x
                ypos[b]=y
                batch[b] = cframe[y:y + imSize, x:x + imSize].astype('float32')/255
                b+=1
                if b==batchSize:
        
                    result = model.predict(batch)
                    positives = result[:,1]#np.argmax(result,axis=1)
                    output[ypos,xpos]=positives
                    b=0
        output = cv2.resize(output[::stepSize,::stepSize],(nx,ny)).astype(np.uint8)
        #outRGB = cv2.cvtColor(255*output,cv2.COLOR_GRAY2BGR)
        #cv2.imwrite('test1.png',outRGB)
        
        labels = np.zeros((ny,nx),dtype=np.uint8)
        lbl, num_features = nd.label(output)
        com = nd.measurements.center_of_mass(output, lbl,np.arange(0,num_features))
        sums = nd.sum(output, lbl, index=np.arange(num_features))
        for feat in range(num_features):
            if sums[feat]<150: 
                continue
            if sums[feat]>400: 
                continue
            rr, cc = draw.circle(com[feat][0], com[feat][1], 6,labels.shape)
            labels[rr,cc]=1
            print(com[feat])

 #       outRGB = cv2.cvtColor(255*labels,cv2.COLOR_GRAY2BGR)
        cv2.imwrite(outputname + '.png',cframe)
        np.save(outputname + '.npy', labels)
        continue
        break
        # first pass rescales to 8x8
        ps1= 8
        rescale = wsize//ps1
        
        impass1 = cv2.resize(frame,(ny//rescale,nx//rescale))
        output = np.zeros_like(impass1)
        exframe = tf.reshape(impass1.astype(np.float32),[1,nx//rescale,ny//rescale,3])
        aa=tf.extract_image_patches(exframe,[1,ps1,ps1,1],[1,1,1,1],[1,1,1,1],'SAME')
        ab=tf.reshape(aa,[nx//rescale*ny//rescale, ps1, ps1,3])
        bigDetect=sess.run(ab)
        NUM = bigDetect.shape[0]
        result = np.zeros((NUM,2))
        
        MAXBLOCK = 1024*256
        
        iters = NUM//MAXBLOCK
        
        for k in range(iters+1):
            
            stop = min((k+1)*MAXBLOCK,NUM)
            
            feed_dict_testing = {x: bigDetect[k*MAXBLOCK:stop,:], y_true: y_test_images}
            result[k*MAXBLOCK:stop,:]=sess.run(y_pred, feed_dict=feed_dict_testing)
        
                    # result is of this format [probabiliy_of_rose probability_of_sunflower]
                    #positives = np.argmax(result,axis=1)
        res2 = result[:,1]
        output = 255.0*np.reshape(res2,(nx//rescale,ny//rescale))
        break
    break
#        
#        #if (tt%60)!=0:
#         #   continue
#     
#        blockSize = 128
#        xblocks = nx//blockSize
#        yblocks = ny//blockSize
#        for xb in range(xblocks):
#            for yb in range(yblocks):
#                print(xb,yb)
#                bx=128
#                by=bx
#                subFrame=frame[:by,:bx,:]
#                exframe = tf.reshape(subFrame.astype(np.float32),[1,by,bx,3])
#                aa=tf.extract_image_patches(exframe,[1,16,16,1],[1,2,2,1],[1,1,1,1],'VALID')
#                
#                ab=tf.reshape(aa,[((by-16)//2+1)*((bx-16)//2+1),16,16,3])
#                h=tf.get_session_handle(ab)
#                h=sess.run(h)
#                feed_dict_testing = {x: h, y_true: y_test_images}
#                result = sess.run(y_pred, feed_dict=feed_dict_testing)
#                #bigDetect=sess.run(ab)
##return nn.forward_prop(X_matrix, Theta1, bias1, Theta2, bias2)
#        
#        def window_func(w):
#    #print(w)
#            ac=tf.reshape(ab[0],[1,16,16,3])
#            bigDetect=sess.run(ac)
#            #h=tf.get_session_handle(ac)
#            #h=sess.run(h)
#            feed_dict_testing = {x: bigDetect, y_true: y_test_images}
#            return 
#        elems = np.array([1, 2, 3, 4, 5, 6])
#        output_t = tf.map_fn(window_func, tf.range(1))
#        output = sess.run(output_t)
#        
#        batchSize=4*1024
#        bigDetect = np.zeros((batchSize,8,8,3))
#        largeF = cv2.resize(frame,(rescale*nx,rescale*ny))
#        
#        batch = 0
#        batchPos = np.zeros((batchSize,2),dtype=int)
#        
#        imSize2 = imSize//2
#        
#       # if tt == 0:
#        for xx in range(border,nx-border,4):
#            #print(xx)
#            for yy in range(border,ny-border,4):
#                ystart = yy*rescale - imSize2
#                ystop = yy*rescale + imSize2
#                xstart = xx*rescale - imSize2
#                xstop = xx*rescale + imSize2
#                bigDetect[batch]= largeF[ystart:ystop,xstart:xstop]
#                batchPos[batch,0]=xx
#                batchPos[batch,1]=yy
#                batch=batch+1
#                if batch%batchSize==0:
#                    batch = 0
#                    ### Creating the feed_dict that is required to be fed to calculate y_pred 
#                    feed_dict_testing = {x: bigDetect, y_true: y_test_images}
#                    result=sess.run(y_pred, feed_dict=feed_dict_testing)
#                    # result is of this format [probabiliy_of_rose probability_of_sunflower]
#                    #positives = np.argmax(result,axis=1)
#                    output[batchPos[:,1],batchPos[:,0]]=255.0*result[:,1]
#                    #print(np.argmax(result,axis=1))
#                        #break
#        if batch>0:
#            feed_dict_testing = {x: bigDetect, y_true: y_test_images}
#            result=sess.run(y_pred, feed_dict=feed_dict_testing)
#            # result is of this format [probabiliy_of_rose probability_of_sunflower]
#            positives = np.argmax(result,axis=1)
#            output[batchPos[:batch,1],batchPos[:batch,0]]=255.0*result[:batch,1]       
#        smallOutput = output[::4,::4] 
#        aa=cv2.resize(smallOutput,(output.shape[1],output.shape[0]))
        #cv2.imwrite('test.png',output)
#        else:
#            blur = cv2.blur(output,(5,5))
#            cv2.imwrite('test.png',output)
#            indList = np.argwhere(blur>1)
#            print(len(indList))
#            for px in indList:
#                
#                xx = px[1]
#                yy = px[0]
#                
#            #for xx in range(border,nx-border,1):
#                
#            #    for yy in range(border,ny-border,1):
#             #       if output[yy,xx]<128.0:
#              #          continue
#                ystart = yy*rescale - 8
#                ystop = yy*rescale + 8
#                xstart = xx*rescale - 8
#                xstop = xx*rescale + 8
#                bigDetect[batch]= largeF[ystart:ystop,xstart:xstop]
#                batchPos[batch,0]=xx
#                batchPos[batch,1]=yy
#                batch=batch+1
#                if batch%batchSize==0:
#                    batch = 0
#                    ### Creating the feed_dict that is required to be fed to calculate y_pred 
#                    feed_dict_testing = {x: bigDetect, y_true: y_test_images}
#                    result=sess.run(y_pred, feed_dict=feed_dict_testing)
#                    # result is of this format [probabiliy_of_rose probability_of_sunflower]
#                    positives = np.argmax(result,axis=1)
#                    output[batchPos[:,1],batchPos[:,0]]=255.0*result[:,1]
#                    #print(np.argmax(result,axis=1))
#                        #break
#            if batch>0:
#                feed_dict_testing = {x: bigDetect, y_true: y_test_images}
#                result=sess.run(y_pred, feed_dict=feed_dict_testing)
#                # result is of this format [probabiliy_of_rose probability_of_sunflower]
#                positives = np.argmax(result,axis=1)
#                output[batchPos[:batch,1],batchPos[:batch,0]]=255.0*result[:batch,1]       
#                
        
        

        
       # outRGB = cv2.cvtColor(output.astype(np.uint8),cv2.COLOR_GRAY2BGR)
      #  out.write(outRGB)
        
       # cv2.imwrite('test.png',output)
        #        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#        cv2.imshow('image', output)
 #       key = cv2.waitKey(1) & 0xFF
 #       if key==27:    # Esc key to stop
  #          break
  #  cv2.destroyAllWindows() 

                

    
            
            
    
    
    
    cap.release()
    out.release()
    
 #   break

#
#
#ims= np.array(['yes/img_0.png','yes/img_36.png','yes/img_45.png','no/img_153.png','no/img_181.png','no/img_253.png'])
#
#image_size=128
#num_channels=3
#images = np.zeros((6,image_size,image_size,num_channels))
#k=0
#for filename in ims:
#    # Reading the image using OpenCV
#    image = cv2.imread(filename)
#    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
#    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
#    images[k]= (image)
#    k=k+1
#images = np.array(images, dtype=np.uint8)
#images = images.astype('float32')
#images = np.multiply(images, 1.0/255.0) 
##The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
#x_batch = images.reshape(6, image_size,image_size,num_channels)


