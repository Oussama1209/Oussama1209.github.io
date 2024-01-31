#### Disclaimer : This code is a working tool for the tuning of the Kalman filter. It is not meant to be used in the final code.
# To use it, you need to add a '.' before the 'utils.data import *' in the kalman file

from utils.data import *
import numpy as np
from kalman_filter import*
# from motion_control import*
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


# Init variables
pos = Robot(position=Point(x=10, y=10), direction= np.pi/2, found = True)
goal = Point(x=150, y=100)
kalman = Kalman(pos)

# Artificial noise variables
speedvar = 5
posvar = 0.125
thetavar = 0.01
dt = 0.1

# Simulation variables
steps=100
memory = []
Cam = False
# Cam = True

################################################################# SIMULATE THE THYMIO TO TEST THE KALMAN FILTER #################################################################
# GO Straight 
inputspeed=np.array([50,50])
for i in range(steps):
    input = Motors(inputspeed[0], inputspeed[1])
    mot_mes = Motors(np.random.normal(inputspeed[0],speedvar), np.random.normal(inputspeed[1],speedvar))

    pos_next, speed = kalman.kalman_filter(input, mot_mes, pos)

    xnext = np.random.normal(pos_next.position.x + speed[0]*np.cos(pos_next.direction)*dt, posvar)
    ynext = np.random.normal(pos_next.position.y + speed[0]*np.sin(pos_next.direction)*dt, posvar)
    thetanext = np.random.normal(pos_next.direction + speed[1]*dt, thetavar)
    pos = Robot(position=Point(x=xnext, y=ynext), direction=thetanext, found = Cam)
    
    memory.append([pos_next.position.x, pos_next.position.y, pos_next.direction])


# Make a curve 
inputspeed=np.array([25,50])
for i in range(steps):
    input = Motors(inputspeed[0], inputspeed[1])
    mot_mes = Motors(np.random.normal(inputspeed[0],speedvar), np.random.normal(inputspeed[1],speedvar))

    pos_next, speed = kalman.kalman_filter(input, mot_mes, pos)

    xnext = np.random.normal(pos_next.position.x + speed[0]*np.cos(pos_next.direction)*dt, posvar)
    ynext = np.random.normal(pos_next.position.y + speed[0]*np.sin(pos_next.direction)*dt, posvar)
    thetanext = np.random.normal(pos_next.direction + speed[1]*dt, thetavar)
    pos = Robot(position=Point(x=xnext, y=ynext), direction=thetanext, found = Cam)
    
    memory.append([pos_next.position.x, pos_next.position.y, pos_next.direction])


# and the other way around
inputspeed=np.array([50,25])
for i in range(steps):
    input = Motors(inputspeed[0], inputspeed[1])
    mot_mes = Motors(np.random.normal(inputspeed[0],speedvar), np.random.normal(inputspeed[1],speedvar))

    pos_next, speed = kalman.kalman_filter(input, mot_mes, pos)

    xnext = np.random.normal(pos_next.position.x + speed[0]*np.cos(pos_next.direction)*dt, posvar)
    ynext = np.random.normal(pos_next.position.y + speed[0]*np.sin(pos_next.direction)*dt, posvar)
    thetanext = np.random.normal(pos_next.direction + speed[1]*dt, thetavar)
    pos = Robot(position=Point(x=xnext, y=ynext), direction=thetanext, found = Cam)
    
    memory.append([pos_next.position.x, pos_next.position.y, pos_next.direction])

# GO Straight 
inputspeed=np.array([50,50])
for i in range(steps):
    input = Motors(inputspeed[0], inputspeed[1])
    mot_mes = Motors(np.random.normal(inputspeed[0],speedvar), np.random.normal(inputspeed[1],speedvar))

    pos_next, speed = kalman.kalman_filter(input, mot_mes, pos)

    xnext = np.random.normal(pos_next.position.x + speed[0]*np.cos(pos_next.direction)*dt, posvar)
    ynext = np.random.normal(pos_next.position.y + speed[0]*np.sin(pos_next.direction)*dt, posvar)
    thetanext = np.random.normal(pos_next.direction + speed[1]*dt, thetavar)
    pos = Robot(position=Point(x=xnext, y=ynext), direction=thetanext, found = Cam)
    
    memory.append([pos_next.position.x, pos_next.position.y, pos_next.direction])


# #Testing Motion Model
# input = np.array([0,0])
# state = np.array([0,0,10,10,0])
# angle_error = 0

# for i in range(10 * steps):
#     robot = Robot(position=Point(x=state[2], y=state[3]), direction=state[4], found = Cam)
#     MotL, MotR, angle_error = controller(robot, goal, 50, angle_error, dt)
    
#     input = np.array([MotL, MotR]) - input
#     state, G = motion_model(state, input, dt)
    
#     memory.append([state[2], state[3], state[4]])


################################################################# PLOT THE RESULTS #################################################################

memory = np.array(memory)
plt.subplots()
# plt.plot(goal.x, goal.y, 'ro')
# plt.title("Position")
plt.plot(memory[:,0],memory[:,1])
plt.xlim([0, 30])
plt.ylim([30, 0])

plt.subplots()
plt.title("Direction")
plt.plot(memory[:,2])
plt.ylim([-np.pi, np.pi])

plt.show()
