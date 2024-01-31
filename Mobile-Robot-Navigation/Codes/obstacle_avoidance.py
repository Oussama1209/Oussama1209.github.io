#local avoidance with obstacle identification via prox sensors
#we used the potential field Navigation

import numpy as np
from .utils.data import *


def obstacle_avoidance(prox_horizontal):
    """
    This function implements obstacle avoidance using the potential field navigation approach.
    It returns the velocity of the robot in the local frame based on the horizontal proximity sensor readings.

    :param prox_horizontal: List of horizontal proximity sensor readings.
    :return: Tuple (obstacle_detected, left_velocity, right_velocity).
             - obstacle_detected: Boolean indicating whether an obstacle is detected.
             - left_velocity: Velocity adjustment for the left motor in the local frame.
             - right_velocity: Velocity adjustment for the right motor in the local frame.
    """
    obstThr = 1500 #detection of the white box = 3 units
    closeobsThr = 3000 #detection of the wall = 2 units
    
    addLeft = 0
    addRight = 0
    weight=[6, 4, -3, -6, -8] #weight of associated to each sensor

    obstacle_detected=False

    prox_horizontal = prox_horizontal[0:5] #we only use the 5 front sensors

    # Check if any sensor reading is above the obstacle detection threshold
    if any(value > obstThr for value in prox_horizontal):
        obstacle_detected=True

    if not(obstacle_detected):
                # No obstacle detected, return zero velocity adjustments
        return obstacle_detected,0,0

    #this part is inspired from the potential field navigation excercise in session 4
    elif obstacle_detected:
        for i in range(5):
            addLeft += prox_horizontal[i] * weight[i] // 200
            addRight += prox_horizontal[i] * weight[4 - i] // 200
        
        return obstacle_detected, addLeft , addRight
    
    else:
       print("error in obstacle avoidance")



def position_temp_obstacle(prox_horizontal, rob : Robot, temp_obstacles):
    """
    Determine the positions of temporary obstacles detected with the front sensors to the thymio

    Modify a list of Points with the positions of obstacles in the coordinate system of thymio in order to display them 

    """
    
    if prox_horizontal[0] > 1700:
        if prox_horizontal[0] > 2700:
            obs = Point(np.sqrt(8)*np.cos(rob.direction-np.pi/4)+rob.position.x, np.sqrt(8)*np.sin(rob.direction-np.pi/4)+rob.position.y)
            temp_obstacles.append(obs)
        else:
            obs = Point(np.sqrt(13)*np.cos(rob.direction-np.pi/3)+rob.position.x, np.sqrt(13)*np.sin(rob.direction-np.pi/3)+rob.position.y)
            temp_obstacles.append(obs)
    elif prox_horizontal[1] > 2200:
        if prox_horizontal[1] > 3200:
            obs = Point(np.sqrt(10)*np.cos(rob.direction-np.pi*0.4)+rob.position.x, np.sqrt(10)*np.sin(rob.direction-np.pi*0.4)+rob.position.y)
            temp_obstacles.append(obs)
        else:
            obs = Point(np.sqrt(17)*np.cos(rob.direction-np.pi*0.42)+rob.position.x, np.sqrt(17)*np.sin(rob.direction-np.pi*0.42)+rob.position.y)
            temp_obstacles.append(obs)
    elif prox_horizontal[2] > 2600:
        if prox_horizontal[0] > 3800:
            obs = Point(np.sqrt(16)*np.cos(rob.direction)+rob.position.x, np.sqrt(16)*np.sin(rob.direction)+rob.position.y)
            temp_obstacles.append(obs)
        else:
            obs = Point(np.sqrt(9)*np.cos(rob.direction)+rob.position.x, np.sqrt(9)*np.sin(rob.direction)+rob.position.y)
            temp_obstacles.append(obs)
    elif prox_horizontal[3] > 2200:
        if prox_horizontal[3] > 3200:
            obs = Point(np.sqrt(10)*np.cos(rob.direction+np.pi*0.4)+rob.position.x, np.sqrt(10)*np.sin(rob.direction+np.pi*0.4)+rob.position.y)
            temp_obstacles.append(obs)
        else:
            obs = Point(np.sqrt(17)*np.cos(rob.direction+np.pi*0.42)+rob.position.x, np.sqrt(17)*np.sin(rob.direction+np.pi*0.42)+rob.position.y)
            temp_obstacles.append(obs)
    elif prox_horizontal[4] > 1700:
        if prox_horizontal[4] > 2700:
            obs = Point(np.sqrt(8)*np.cos(rob.direction+np.pi/4)+rob.position.x, np.sqrt(8)*np.sin(rob.direction+np.pi/4)+rob.position.y)
            temp_obstacles.append(obs)
        else:
            obs = Point(np.sqrt(13)*np.cos(rob.direction+np.pi/3)+rob.position.x, np.sqrt(13)*np.sin(rob.direction+np.pi/3)+rob.position.y)
            temp_obstacles.append(obs)
    
    

