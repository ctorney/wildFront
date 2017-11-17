
# Force-based interaction model

import numpy as np
from math import *
import matplotlib.pyplot as plt
import sys
import random

random.seed(0)
np.random.seed(0)
N =2

positions = 0.45 + np.random.uniform(0,40,(N,2))
angles = np.random.uniform(0,0.02*pi,N)
speeds = np.random.uniform(0,1,N)
forces  = np.zeros(N)

i=0



ca = 1
cr = 20
lr = 3
la = 5
ka = 0.01

dt = 0.1
gamma = 0.5
D = 0.01
epsilon = sqrt(2*D*dt)

pl = 1


def updatePositions():
    
    for i in range(N):
        forces[i]=0
        dx = 0
        dy = 0
        weights = 0
        nearest = 100
        nearestButt = 100
        nj=-1
        njb=-1
        
        
        for j in range(N):
        
            if i==j:
                continue
            dxj = positions[j,0]-positions[i,0]
            dyj = positions[j,1]-positions[i,1]
            d = sqrt(dxj**2+dyj**2)
            
            
            
        
            dotp = (dxj/d)*(cos(angles[i])) + (dyj/d)*sin(angles[i])
            #print(dotp)
            if dotp<0:
                if d<nearestButt:
             #       print(d)
                    nearestButt=d
                    njb=j
            #else:
            if d<nearest:
                nearest=d
                nj=j                    
    #            weightj = d**pl
    #            weights += weightj
            dx += dxj#*weightj
            dy += dyj#*weightj
#            
#        
        dx = dx / N
        dy = dy / N
#        d = sqrt(dx**2+dy**2)
#        
#        
#        if v>0:
#            dotp = (dx/d)*(vx/v) + (dy/d)*(vy/v)
#        else:
#            dotp = 0
#        
#        cri = cr*(1.0-dotp)/2
#        cai = ca*(dotp+1.0)/2
#        
#        dfadd = 1.0/(1.0+exp(-(d-la)/ka))
#        forces[i,0]+= -cri*dx*exp(-d/lr)/(lr*d) + cai*dx*dfadd/d
#        forces[i,1]+= -cri*dy*exp(-d/lr)/(lr*d) + cai*dy*dfadd/d
        #print(i,nearestButt,nearest)
        buttKick = 10
        if nearestButt<lr:
            dxj = positions[njb,0]-positions[i,0]
            dyj = positions[njb,1]-positions[i,1]
            d = sqrt(dxj**2+dyj**2)
            #angles[i] = atan2(-dyj,-dxj)
            forces[i] = buttKick#*dxj/d
            #forces[i,1] -= buttKick*dyj/d
            print('butt',i)
            continue     
        attract=0.02
        #print(i,nearest)
        d = sqrt(dx**2+dy**2)
        if d>la:
            #dxj = positions[nj,0]-positions[i,0]
            #dyj = positions[nj,1]-positions[i,1]
            dotp = (dx/d)*(cos(angles[i])) + (dy/d)*sin(angles[i])
            if dotp>0:
                angles[i] = atan2(attract*dy/d+sin(angles[i]),attract*dx/d+cos(angles[i]))
            else:
                forces[i]-=10
            #forces[i,0] += attract*dx/d
            #forces[i,1] += attract*dy/d
            #print('attract',i)
              
            
        kicker=random.uniform(0,10)
        kick=0.1
        if random.uniform(0,1)<kick:
            forces[i] += kicker
            #forces[i,1] += kicker*vy/v
        


    
    speeds[:] = speeds[:]  + dt * forces - dt * speeds[:] * gamma 
    speeds[speeds<0]=0 
    #randvel =  np.exp(epsilon*np.random.normal(0,1,N))
    #velocity[:,0] *= randvel#*np.divide(velocity[:,0],(np.linalg.norm(velocity,axis=1)))
    #velocity[:,1] *= randvel#*np.divide(velocity[:,1],(np.linalg.norm(velocity,axis=1)))
    
    positions[:,0] = positions[:,0] + dt * speeds * np.cos(angles)
    positions[:,1] = positions[:,1] + dt * speeds * np.sin(angles)
    
   

# run for this many time steps
TIMESTEPS = 20000

# simulate individual movement
for t in range(TIMESTEPS):
    # individuals move in direction defined by heading with fixed speed
    
    updatePositions()
    # boundary conditions are periodic
    #xpos[xpos<0]=xpos[xpos<0]+1
    #xpos[xpos>1]=xpos[xpos>1]-1
    #ypos[ypos<0]=ypos[ypos<0]+1
    #ypos[ypos>1]=ypos[ypos>1]-1
    if t%10==0:
        # plot the positions of all individuals
        plt.clf()
        #velx = np.divide(velocity[:,0],(np.linalg.norm(velocity,axis=1)))
        #vely = np.divide(velocity[:,1],(np.linalg.norm(velocity,axis=1)))
        plt.quiver(positions[:,0], positions[:,1],np.cos(angles), np.sin(angles))
        plt.axes().set_aspect('equal')
        plt.axis([-200,200,-200,200])
        plt.draw()
        plt.pause(0.001)
        