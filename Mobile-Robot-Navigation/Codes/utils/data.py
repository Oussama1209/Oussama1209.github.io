##########################################################################################################################################
###################### Description: Definition of Datastructure used in this project     #################################################
##########################################################################################################################################


import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
# Source : https://docs.python.org/fr/3/library/dataclasses.html#module-dataclasses

@dataclass
class Point:      
    """ Object to caracterize a position """
    x : float
    y : float
    
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y]) 

    def dist(self, other) -> float:
        return np.linalg.norm(self.pos() - other.pos())
    
    def copy(self):
        return Point(self.x, self.y)
    
    def update(self, newPoint):
        self.x = newPoint.x
        self.y = newPoint.y

@dataclass
class Map:
    """ Object to caracterize a map """
    corners : list[Point]
    obstacles : list[Point]
    frame : Optional[np.ndarray] = None   # frame of the map (for display)

    def update(self, newMap, frame=None):
        self.corners = newMap.corners
        self.obstacles = newMap.obstacles
        self.frame = frame

@dataclass
class Robot:
    """ Object to caracterize the Thymio in space """
    position : Point
    direction : float       #angle from x direction (->) 
    found : Optional[bool] = False  # False if can't see the robot

    def copy(self):
        return Robot(self.position.copy(), self.direction, self.covariance.copy())
    
    def update(self, newRobot):
        self.position = newRobot.position
        self.direction = newRobot.direction
        self.found = newRobot.found
        
@dataclass
class Environment:
    """ Object to caracterize the full environment """
    robot : Robot
    map   : Map
    goal  : Point

    def update(self, newEnv):
        self.robot = newEnv.robot
        self.map = newEnv.map
        self.goal = newEnv.goal

@dataclass
class Motors:
    """ Object to describe Thymio's motor speeds """
    left  : int = 0
    right : int = 0

    def update(self, newMotors):
        self.left = newMotors.left
        self.right = newMotors.right


@dataclass
class Sensors:
    """ Object to regroup Thymio's prox & ground sensors """
    prox   : list[int]   # 7 int list
    ground : list[int]   # 2 int list

    def left(self):
        return self.prox[0]
    
    def center(self):
        return self.prox[2]

    def right(self):
        return self.prox[4]
    
    def behind_l(self):
        return self.prox[5]
    
    def behind_r(self):
        return self.prox[6]
    

@dataclass 
class Lights:
    """ Object to caracterize Thymio's top lights """
    color : list[int]  # 3 int list

@dataclass 
class Camera:
    """ Object to caracterize Camera """
    Status : Optional[bool] = False  # ON = 1, OFF = 0

    def update(self, status):
        self.Status = status