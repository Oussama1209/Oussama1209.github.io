##########################################################################################################################################
################################################# Description: Test file for multithreading ##############################################
##########################################################################################################################################

## Lib import
from timeit import default_timer as timer
from tdmclient import ClientAsync, aw
import time
import cv2 as cv
import threading
import keyboard

from Codes.utils.data import *
from Codes.utils.communication import *
from Codes.kalman_filter import *
from Codes.motion_control import *
from Codes.obstacle_avoidance import *
from Codes.camera import *
from Codes.map import *
 

## Constants
GRID_RES = 25
TIMESTEP = 0.1
SIZE_THYM = 2.0  #size of thymio in number of grid
LOST_TRESH = 10 #treshold to be considered lost
REACH_TRESH = 3 #treshhold to reach current checkpoint
GLOBAL_PLANNING = True
GOAL_REACHED = False


## Functions

def run_cam(mes_pos):
    '''
    Function that updates the global Mes_Robot variable with camera data every 0.1 seconds

    '''
    i = 0 
    last = 0
    print("Camera thread started")
    while True:
        print("Camera update")
        print(time.time() - last)
        last = time.time()
        time.sleep(0.1)

def update_thym(thymio):
    '''
    Function that updates the variables of the thymio object (sensor data) every 0.1 seconds

    '''
    i = 0
    last = 0
    print("Thymio thread started")
    while True:
        print("Thymio update")
        print(time.time() - last)
        last = time.time()
        time.sleep(0.1)


## Main

if __name__ == "__main__":
    
    print("Main Thread started")

    # Init variables
    thymio = Motors(0, 0)
    Mes_car = Robot(Point(0, 0), 0)
    input = Motors(0, 0 )

    # Launch Threads
    camera_thread = threading.Thread(target=run_cam, args=(Mes_car,), daemon=True)
    camera_thread.start()

    thymio_thread = threading.Thread(target=update_thym, args=(thymio,), daemon=True)
    thymio_thread.start()


    # Init timer
    start = time.time()
    current = start

    # Init loop
    while True:
        # Update motion every 0.1 seconds
        deltaT = current - start
        if deltaT < TIMESTEP:
            time.sleep(0.01)
            current = time.time()
            continue

        print("Main motion control")
        print(current - start)


        # Check if escape key pressed
        if keyboard.is_pressed('esc'):
            break

        # Update timer
        start = time.time()
        current = start


