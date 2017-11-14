
# Force-based interaction model

import numpy as np
from math import *
import matplotlib.pyplot as plt
import sys


N = 10

positions = 0.45 + np.random.uniform(0,0.1,(N,2))
velocity = np.zeros((N,2))
forces  = np.zeros((N,2))

i=0



ca = 1
cr = 2
lr = 0.01
la = 0.5
ka = 5

dt = 0.01
gamma = 1
D = 0.1

epsilon = sqrt(2*D*dt)

def updateSocial():
    
    for i in range(N):
        forces[i,:]=0
        for j in range(N):
        
            if i==j:
                continue
            dx = positions[j,0]-positions[i,0]
            dy = positions[j,1]-positions[i,1]
            d = sqrt(dx**2+dy**2)
            
            forces[i,0]+= -cr*dx*exp(-d/lr)/(lr*d) + ca*dx*exp(-d/la)/(la*d)
            forces[i,1]+= -cr*dy*exp(-d/lr)/(lr*d) + ca*dy*exp(-d/la)/(la*d)


def updatePositions():
    
    velocity[:] = velocity[:] - dt * velocity[:] * gamma + dt * forces + epsilon*np.random.normal(0,1,(N,2))
    positions[:] = positions[:] + dt * velocity
    
   
# set to random initial conditions on (0,1)x(0,1)
xpos = np.random.uniform(0,1,N)
ypos = np.random.uniform(0,1,N)

# set to random inital headings
heading = np.random.uniform(0,2*pi,N)

# set speed individuals move
speed = 0.01

# run for this many time steps
TIMESTEPS = 200

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
    if t%1==0:
        # plot the positions of all individuals
        plt.clf()
        plt.plot(positions[:,0], positions[:,1],'k.')
        plt.axes().set_aspect('equal')
        plt.axis([0,1,0,1])
        plt.draw()
        plt.pause(0.01)
        