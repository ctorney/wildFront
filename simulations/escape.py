
# Force-based interaction model

import numpy as np
from math import *
import matplotlib.pyplot as plt
import sys
import random

random.seed(10)
np.random.seed(10)
N =20
sz = 200

# what is the point of taking own movement into account
# would it be the same if neighbours were really approaching or retreating
lm=10
la=5
lc=2.0
muc=20
mua=-3
mum=3
dt = 0.05
D = 0.01
epsilon = sqrt(2*D*dt)
DV = 0.25
epsV = sqrt(2*DV*dt)
gamma = 1

alpha = gamma+DV
vs=2.0*alpha/gamma


positions = np.random.uniform(sz/2,sz/2+sqrt(N)*2*lc,(N,2))
angles = np.random.uniform(0,2*pi,N)
speeds = vs*np.ones(N)
Fi_theta  = np.zeros(N)

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
        coll=1.0
        for j in range(N):
        
            if i==j:
                continue
            dxj = positions[j,0]-positions[i,0]
            dyj = positions[j,1]-positions[i,1]
            if dxj>0.5*sz:
                dxj-=sz
            if dxj<-0.5*sz:
                dxj+=sz
            if dyj>0.5*sz:
                dyj-=sz
            if dyj<-0.5*sz:
                dyj+=sz                
            d = sqrt(dxj**2+dyj**2)
            dxj/=d
            dyj/=d
            if d<lc:
                fc_x += (dxj)
                fc_y += (dyj)
                fc_count+=1
            else:
                if d>la and d>lm:
                    continue
                vx = speeds[j]*cos(angles[j])-speeds[i]*cos(angles[i])
                vy = speeds[j]*sin(angles[j])-speeds[i]*sin(angles[i])
                #vx = cos(angles[j])-cos(angles[i])
                #vy = sin(angles[j])-sin(angles[i])
                vhat = vx*dxj + vy*dyj
                if vhat<0 and d<la:
                    fa_x += -vhat*dxj
                    fa_y += -vhat*dyj
                    fa_count += 1
                elif d<lm:
                    fm_x += vhat*dxj
                    fm_y += vhat*dyj
                    fm_count += 1
                
        if fc_count:
            fc_x = -muc*fc_x/fc_count
            fc_y = -muc*fc_y/fc_count
            coll=5.0
            
        if fa_count:
            fa_x = mua*fa_x/fa_count
            fa_y = mua*fa_y/fa_count
            
        if fm_count:
            fm_x = mum*fm_x/fm_count
            fm_y = mum*fm_y/fm_count
        
        Fx = fc_x+fm_x+fa_x
        Fy = fc_y+fm_y+fa_y
        
        Fi_theta[i] = -Fx*sin(angles[i])+Fy*cos(angles[i])
        
    #print('======')
    #speeds[:] = speeds[:]  + dt * forces - dt * speeds[:] * gamma 
    
    
    angles[:] = angles[:] + ((dt * Fi_theta[:]) + epsilon*np.random.normal(0,1,N))#/(0.5+0.5*speeds[:])
    #speeds[:] = speeds[:] + ((dt * gamma * (vs-speeds[:]))) + (epsV*np.random.normal(0,1,N))*speeds[:]
    angles[0]=0
    speeds[:] = speeds[:]*np.exp((epsV*np.random.normal(0,1,N))-alpha*dt) + (1.0-np.exp(-alpha*dt))*gamma*vs*coll/alpha
    #speeds[speeds<0]=0 
    #randvel =  np.exp()
    #velocity[:,0] *= randvel#*np.divide(velocity[:,0],(np.linalg.norm(velocity,axis=1)))
    #velocity[:,1] *= randvel#*np.divide(velocity[:,1],(np.linalg.norm(velocity,axis=1)))
    angles[angles<-pi]=angles[angles<-pi]+2*pi
    angles[angles>pi]=angles[angles>pi]-2*pi
    positions[:,0] = positions[:,0] + dt * speeds * np.cos(angles)
    positions[:,1] = positions[:,1] + dt * speeds * np.sin(angles)
    
   

# run for this many time steps
TIMESTEPS = 2000

sp1 = np.zeros(TIMESTEPS)
# simulate individual movement
for t in range(TIMESTEPS):
    # individuals move in direction defined by heading with fixed speed
  #  print(t)
    updatePositions()
    # boundary conditions are periodic

    positions[positions[:,0]<0,0]=positions[positions[:,0]<0,0]+sz
    positions[positions[:,0]>sz,0]=positions[positions[:,0]>sz,0]-sz
    positions[positions[:,1]<0,1]=positions[positions[:,1]<0,1]+sz
    positions[positions[:,1]>sz,1]=positions[positions[:,1]>sz,1]-sz
    
    sp1[t]=speeds[0]
    #xpos[xpos>1]=xpos[xpos>1]-1
    #ypos[ypos<0]=ypos[ypos<0]+1
    #ypos[ypos>1]=ypos[ypos>1]-1
    if t%10==0:
        # plot the positions of all individuals
        plt.clf()
        plt.quiver(positions[:,0], positions[:,1],np.cos(angles), np.sin(angles))
        plt.axes().set_aspect('equal')
        plt.axis([0,sz,0,sz])
        plt.draw()
        plt.pause(0.001)
plt.figure()
plt.plot(sp1)
#print(np.mean(sp1),gamma*vs/(gamma+D))