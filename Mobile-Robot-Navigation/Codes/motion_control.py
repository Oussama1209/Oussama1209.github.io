import math
import numpy as np
from .utils.data import *

# PD Constants
Kp = 100/np.pi
Kd = Kp * 0.1
# Kd = 0

def compute_angle_error(position, checkpoint, thymio_direction):
  """
  Computes the angle error between the robot and the goal.

  :param position: Current position of the robot.
  :param checkpoint: position to reach (next point in the global path).
  :param thymio_angle: Current orientation of the robot.
  :return: Angle error between the robot and the goal.
  """
  goal_angle = np.arctan2( (checkpoint.y - position.y), (checkpoint.x - position.x))

  thymio_direction = - thymio_direction    # Because the y-axis is inverted in the simulator  

  angle_error = goal_angle - thymio_direction         
  angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

  return angle_error

def compute_derived_angle_error(angle_error, prev_angle_error, dt):
  """
  Computes the derived angle error between the robot and the goal.

  :param angle_error: Angle error between the robot and the goal.
  :param prev_angle_error: Previous angle error between the robot and the goal.
  :param dt: Time step.
  :return: Derived angle error between the robot and the goal.
  """
  return (angle_error - prev_angle_error) / dt


def controller(robot, checkpoint, base_speed, prev_angle_error, dt, final_goal_reached=False):
  '''
  This function computes the motor velocities based on the position, goal, and control parameters.
  param robot : Current position of the robot
  param checkpoint : Goal position to reach (next point in the global path)
  param base_speed : Base speed of the robot
  param final_goal_reached : Flag indicating whether the final goal is reached, not motion is needed.

  return : Input as a tuple (MotL, MotR) 
  return : angle_error
  '''
  if final_goal_reached:
      return 0, 0, 0
  
  angle_error = compute_angle_error(robot.position, checkpoint, robot.direction)
  derived_angle_error = compute_derived_angle_error(angle_error, prev_angle_error, dt)

  rot_speed = Kp * angle_error + Kd * derived_angle_error

  # Clip the rotation speed to avoid saturation
  rot_speed = rot_speed if abs(rot_speed) < 200 - base_speed else 200 - base_speed

  MotL = base_speed + rot_speed
  MotR = base_speed - rot_speed

  print("MotL: ", MotL, "MotR: ", MotR)
  
  return MotL, MotR, angle_error
  # return 0, 0, angle_error



