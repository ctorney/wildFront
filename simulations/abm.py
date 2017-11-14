
# Force-based interaction model

import numpy as np
from math import *
import matplotlib.pyplot as plt
import sys


N = 100

positions = 0.45 + np.random.uniform(0,1,(N,2))
velocity = np.random.uniform(-1,1,(N,2))
forces  = np.zeros((N,2))

i=0



ca = 1
cr = 2
lr = 0.01
la = 0.5
ka = 0.5

dt = 0.01
gamma = 1
D = 0.1
epsilon = sqrt(2*D*dt)

pl = 1

def updateSocial():
    
    for i in range(N):
        forces[i,:]=0
        dx = 0
        dy = 0
        weights = 0
        for j in range(N):
        
            if i==j:
                continue
            dxj = positions[j,0]-positions[i,0]
            dyj = positions[j,1]-positions[i,1]
            d = sqrt(dxj**2+dyj**2)
            
            weightj = d**pl
            weights += weightj
            dx += dxj*weightj
            dy += dyj*weightj
        
        dx = dx / weights
        dy = dy / weights
        d = sqrt(dx**2+dy**2)
        vx = velocity[i,0]
        vy = velocity[i,1]
        
        v = sqrt(vx**2+vy**2)
        if v>0:
            dotp = (dx/d)*(vx/v) + (dy/d)*(vy/v)
        else:
            dotp = 0
        
        cri = cr*(1.0-dotp)/2
        cai = ca*(dotp+1.0)/2
        
        dfadd = 1.0/(1.0+exp(-(d-la)/ka))
        forces[i,0]+= -cri*dx*exp(-d/lr)/(lr*d) + cai*dx*dfadd/d
        forces[i,1]+= -cri*dy*exp(-d/lr)/(lr*d) + cai*dy*dfadd/d
        
        


def updatePositions():
    
    velocity[:] = velocity[:]  + dt * forces - dt * velocity[:] * gamma#+ epsilon*np.random.normal(0,1,(N,2))
    #velocity[:,0] = np.divide(velocity[:,0],(np.linalg.norm(velocity,axis=1)))
    #velocity[:,1] = np.divide(velocity[:,1],(np.linalg.norm(velocity,axis=1)))
    
    positions[:] = positions[:] + dt * velocity
    
   

# run for this many time steps
TIMESTEPS = 2000

# simulate individual movement
for t in range(TIMESTEPS):
    # individuals move in direction defined by heading with fixed speed
    updateSocial()
    updatePositions()
    # boundary conditions are periodic
    #xpos[xpos<0]=xpos[xpos<0]+1
    #xpos[xpos>1]=xpos[xpos>1]-1
    #ypos[ypos<0]=ypos[ypos<0]+1
    #ypos[ypos>1]=ypos[ypos>1]-1
    if t%10==0:
        # plot the positions of all individuals
        plt.clf()
        plt.plot(positions[:,0], positions[:,1],'k.')
        plt.axes().set_aspect('equal')
        plt.axis([-10,10,-10,10])
        plt.draw()
        plt.pause(0.001)
        