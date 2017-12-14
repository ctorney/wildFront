
import numpy as np
import cv2
import glob
import itertools
import random




def generate(train_path, batch_size  , nx , ny):
    
    assert train_path[-1] == '/'
    
    n_class=2
    images = glob.glob( train_path + "*.png"  ) 
    images.sort()
    segmentations  = glob.glob( train_path + "*.npy"  ) 
    segmentations.sort()

    assert len( images ) == len(segmentations)

    pair_list = list(zip(images,segmentations))
    random.shuffle(pair_list)
    zipped = itertools.cycle(pair_list  )

    while True:
        X = []
        Y = []
        for _ in range( batch_size) :
            imname,segname = next(zipped)
            img = cv2.imread(imname, 1)
            minD = min(img.shape[0],img.shape[1])
            offset = random.randint(0,minD-nx)
            label = np.load(segname)
            
            
            img = img[offset:offset+ny,offset:offset+nx,:]
            label = label[offset:offset+ny,offset:offset+nx]
            
            train_labels = np.zeros((label.shape[0],label.shape[1],n_class))
            
            
            for c in range(n_class):
                train_labels[: , : , c ] = (label == c ).astype(int)

                        
            X.append( img.astype('float32')/255  )
            Y.append( train_labels )

        yield np.array(X) , np.array(Y)


