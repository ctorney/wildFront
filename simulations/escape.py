
# Force-based interaction model

import numpy as np
from math import *
import matplotlib.pyplot as plt
import sys
import random

random.seed(0)
np.random.seed(0)
N =200
sz = 200

vs=1.0
ls=5
lc=1.0
muc=20
mua=-3
mum=3
dt = 0.1
D = 0.01
epsilon = sqrt(2*D*dt)/vs


positions = np.random.uniform(sz/2,sz/2+sqrt(N)*2*lc,(N,2))
angles = np.random.uniform(0,2*pi,N)
speeds = vs*np.ones(N)
forces  = np.zeros(N)

def heavy(x,y):
    if x>y:
        return 1
    else:
        return 0
        
def updatePositions():
    
    for i in range(N):
        Fx=0
        Fy=0
        fc_x = 0
        fc_y = 0
        fm_x = 0
        fm_y = 0
        fa_x = 0
        fa_y = 0
        fc_count = 0
        fm_count = 0
        fa_count = 0
        for j in range(N):
        
            if i==j:
                continue
            dxj = positions[j,0]-positions[i,0]
            dyj = positions[j,1]-positions[i,1]
            d = sqrt(dxj**2+dyj**2)
            dxj/=d
            dyj/=d
            if d<lc:
                fc_x += (dxj)
                fc_y += (dyj)
                fc_count+=1
            else:
                if d>ls:
                    continue
                vx = cos(angles[j])-cos(angles[i])
                vy = sin(angles[j])-sin(angles[i])
                vhat = vx*dxj + vy*dyj
                if vhat<0:
                    fa_x += -vhat*dxj
                    fa_y += -vhat*dyj
                    fa_count += 1
                else:
                    fm_x += vhat*dxj
                    fm_y += vhat*dyj
                    fm_count += 1
                
        if fc_count:
            fc_x = -muc*fc_x/fc_count
            fc_y = -muc*fc_y/fc_count
            
        if fa_count:
            fa_x = mua*fa_x/fa_count
            fa_y = mua*fa_y/fa_count
            
        if fm_count:
            fm_x = mum*fm_x/fm_count
            fm_y = mum*fm_y/fm_count
        
        Fx = fc_x+fm_x+fa_x
        Fy = fc_y+fm_y+fa_y
        
        Fi_theta = -Fx*sin(angles[i])+Fy*cos(angles[i])

    #print('======')
    #speeds[:] = speeds[:]  + dt * forces - dt * speeds[:] * gamma 
    #speeds[speeds<0]=0 
    
        angles[i] = angles[i] + (dt * Fi_theta/vs) + epsilon*np.random.normal(0,1,1)
    #randvel =  np.exp()
    #velocity[:,0] *= randvel#*np.divide(velocity[:,0],(np.linalg.norm(velocity,axis=1)))
    #velocity[:,1] *= randvel#*np.divide(velocity[:,1],(np.linalg.norm(velocity,axis=1)))
    angles[angles<-pi]=angles[angles<-pi]+2*pi
    angles[angles>pi]=angles[angles>pi]-2*pi
    positions[:,0] = positions[:,0] + dt * speeds * np.cos(angles)
    positions[:,1] = positions[:,1] + dt * speeds * np.sin(angles)
    
   

# run for this many time steps
TIMESTEPS = 20000

# simulate individual movement
for t in range(TIMESTEPS):
    # individuals move in direction defined by heading with fixed speed
    
    updatePositions()
    # boundary conditions are periodic

    positions[positions[:,0]<0,0]=positions[positions[:,0]<0,0]+sz
    positions[positions[:,0]>sz,0]=positions[positions[:,0]>sz,0]-sz
    positions[positions[:,1]<0,1]=positions[positions[:,1]<0,1]+sz
    positions[positions[:,1]>sz,1]=positions[positions[:,1]>sz,1]-sz
    
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
        plt.axis([0,sz,0,sz])
        plt.draw()
        plt.pause(0.001)
        