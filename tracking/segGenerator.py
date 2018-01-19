
import numpy as np
import cv2
import glob
import itertools
import random




def generate(train_path, batch_size  , nx , ny):
    
    assert train_path[-1] == '/'
    
    n_class=2
    images = glob.glob( train_path + "*.png"  ) 
 #   images = glob.glob( train_path + "*MOVIE-6-1488-0-0.png"  ) + glob.glob( train_path + "*MOVIE-0-00-0.png"  ) 
    images.sort()
    segmentations  = glob.glob( train_path + "*.npy"  ) #+ glob.glob( train_path + "*MOVIE-0-00-0.npy"  ) 
    segmentations.sort()

    assert len( images ) == len(segmentations)

    pair_list = list(zip(images,segmentations))
    random.shuffle(pair_list)
    zipped = itertools.cycle(pair_list  )

    while True:
        X = []
        Y = []
        b=0
        while b < batch_size:
 #       for _ in range( batch_size) :
            imname,segname = next(zipped)
            img = cv2.imread(imname, 1)
            minD = min(img.shape[0],img.shape[1])
            offset = random.randint(0,minD-nx)
            label = np.load(segname)
            
            
            img = img[offset:offset+ny,offset:offset+nx,:]
            label = label[offset:offset+ny,offset:offset+nx]
            if np.sum(label)==0:
                if random.uniform(0,1)>0.01:
                    continue

            
            train_labels = np.zeros((label.shape[0],label.shape[1],n_class))
            
            
            for c in range(n_class):
                if c==1:
                    sf = 100
                else:
                    sf = 1

                train_labels[: , : , c ] = (label == c ).astype(int)*sf

                        
            b = b + 1
            X.append( img.astype('float32')/255  )
            Y.append( train_labels )

        yield np.array(X) , np.array(Y)


