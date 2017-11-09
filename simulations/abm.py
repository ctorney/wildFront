
# Force-based interaction model

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import sys


N = 3

xpos = np.random.uniform(0,1,N)
ypos = np.random.uniform(0,1,N)

i=0

xpos[0]=0.5
ypos[0]=0.5


sumX=0
for j in range(N):
    if i==j:
        sumX = 
        

sys.exit('bye!')

# number of individuals
N=100

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
    xpos = xpos + speed*np.cos(heading)
    ypos = ypos + speed*np.sin(heading)
    
    # boundary conditions are periodic
    xpos[xpos<0]=xpos[xpos<0]+1
    xpos[xpos>1]=xpos[xpos>1]-1
    ypos[ypos<0]=ypos[ypos<0]+1
    ypos[ypos>1]=ypos[ypos>1]-1

    # plot the positions of all individuals
    plt.clf()
    plt.plot(xpos, ypos,'k.')
    plt.axes().set_aspect('equal')
    plt.draw()
    plt.pause(0.01)
    