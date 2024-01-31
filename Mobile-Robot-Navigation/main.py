##########################################################################################################################################
############################################# Description: Main file of the project  #####################################################
##########################################################################################################################################

## Disclaimer : The code may need some adjustement to work on your computer.
#   - the camera may not be recognized as the first camera of the computer
#   - the black and white threshold may need to be adjusted

## Usage : In order to quit the code once launched you need to press the 'esc' key on your keyboard and 'q' in the camera windows


## Lib import
from timeit import default_timer as timer
from tdmclient import ClientAsync, aw
import time
import sys
import cv2 as cv
import threading
import keyboard

from Codes.utils.data import *
from Codes.utils.communication import *
from Codes.kalman_filter import *
from Codes.motion_control import *
from Codes.obstacle_avoidance import *
from Codes.vision import *
from Codes.map import *
 

## Constants
DEFAULT_GRID_RES = 19 #resolution of the grid in pixels
TIMESTEP = 0.075 #time between each iteration of the main loop
SIZE_THYM = 2.5  #size of thymio in number of grid
SPEEDCONV = 0.05
BASESPEED = 50
LOST_TRESH = 6   #treshold to be considered lost
REACH_TRESH = 3  #treshhold to reach current checkpoint
REACH_GOAL_TRESH = 1 #treshhold to reach final goal
GLOBAL_PLANNING = True
GOAL_REACHED = False
GO_NEXT = False

#####################################################################################################################
#####################################################################################################################
## Threads Functions

def run_camera(mes_pos : Robot, mes_goal: Point, camera : Camera, grid_res=DEFAULT_GRID_RES):
    '''
    Function that updates the global Mes_Robot variable with camera data every 0.1 seconds on average

    '''
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    camera.update(ret)
    
    width, height = 707, 10007
    # width, height = set_param_frame(arucos)

    last_known_goal_pos = (0, 0)
    robot_pos = (0, 0)
    while True:
        if cap.isOpened() == False:
            print("Error : video stream closed")
        else:
            frame = cap.read()[1]

            frame, arucos = get_arucos(frame)

            frame, arucos, robot_pos, angle = show_robot(frame, arucos, grid_res)
            goal_pos = center_in_grid(arucos, 99, grid_res)

            if robot_pos != (0,0) :
                robot_pos = Point(robot_pos[0], robot_pos[1])
                mes_pos.update(Robot(robot_pos, angle, True))
            else:
                mes_pos.update(Robot(mes_pos.position, mes_pos.direction, False))
                    
            if goal_pos != (0, 0):
                mes_goal.update(Point(goal_pos[0], goal_pos[1]))
            draw_goal(frame, arucos, grid_res)

            # frame = projected_image(frame, arucos, width, height)
            cv2.imshow("Video Stream", frame)

            # print(f'Robot position: {robot_pos} and angle: {angle}')
            # print(f'Goal position: {goal_pos}')
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        time.sleep(0.095)

#####################################################################################################################

def update_thymio(thymio : Thymio):
    '''
    Function that updates the variables of the thymio object (sensor data) every 0.1 seconds on average

    '''
    while True:
        thymio.read_variables()
        time.sleep(0.075)

#####################################################################################################################

def display(env : Environment, path : list, visitedNodes : list, map : Map, grid_res=DEFAULT_GRID_RES):
    '''
    Function that displays the current state of the robot and the goal on the screen
    '''
    print("Starting display thread...")

    # Update display
    while True:
        display = cv.imread(".\captured_frame.png")
        if env.map.frame is not None:     #check if frame is not empty
                
                display = show_grid(display, grid_res)

                #show goal position with red point
                display = draw_circle(display, (env.goal.x, env.goal.y), grid_res, radius=10, color=(0, 0, 255), thickness=-1)
            
                #show robot position with green point + arrow
                circle_pos = (int(env.robot.position.x*grid_res)/grid_res, int(env.robot.position.y*grid_res)/grid_res)
                display = draw_circle(display, circle_pos, grid_res, radius=5, color=(0, 255, 0), thickness=-1)
                display = draw_arrow_from_robot(display, env.robot, grid_res)

                #show obstacles with grey points
                for obs in env.map.obstacles:
                    display = draw_circle(display, (obs.x, obs.y), grid_res, radius=2, color=(100, 100, 100), thickness=-1)

                for point in extended_obs:
                    display = draw_circle(display, (point.x, point.y), grid_res, radius=2, color=(30, 30, 30), thickness=-1)

                if path != []:
                    #show path with blue points 
                    for point in path:
                        display = draw_circle(display, (point.x, point.y), grid_res, radius=3, color=(255, 0, 0), thickness=-1) 
                else:
                    #show visited nodes with yellow points 
                    for point in visitedNodes:
                        display = draw_circle(display, (point.x, point.y), grid_res, radius=2, color=(0, 200, 200), thickness=-1)
                
                #show temporary obstacles detected with proximity sensors
                for point in temp_obstacles:
                    display = display = draw_circle(display, (point.x, point.y), grid_res, radius=2, color=(0, 100, 200), thickness=-1)

                    
        cv.imshow("Positions", display) 

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        time.sleep(0.2)

#####################################################################################################################
#####################################################################################################################
## Main

if __name__ == "__main__":
    # Init Thymio
    thymio = Thymio()

    # Init map
    map = Map([], [], None)

    builtmap, grid_res = apply_grid_to_camera(DEFAULT_GRID_RES)
    frame = cv2.imread("captured_frame.png")
    
    map.update(builtmap, frame)

    # Init variables
    Mes_car = Robot(Point(0,0), 0)  # Mes_car = measured car
    Mes_goal = Point(0,0)           # Mes_goal = measured goal
    theta_err = 0                   # prev_angle_error = previouses angle errors
    env = Environment(Mes_car, map, Mes_goal)       # env = estimated environment (updated with Kalman)
    input = Motors(0,0)             # input = motors command
    path = list()                   # path = list of checkpoints to reach
    visitedNodes = list()           # visitedNodes = list of visited nodes during A* algorithm
    camera = Camera()
    extended_obs = list()           # extended_obs = list of obstacles with extended size of the robot
    temp_obstacles = list()

    # Launch Threads
    camera_thread = threading.Thread(target=run_camera, args=(Mes_car, Mes_goal, camera, grid_res), daemon=True)
    camera_thread.start()

    thymio_thread = threading.Thread(target=update_thymio, args=(thymio,), daemon=True)
    thymio_thread.start()

    display_thread = threading.Thread(target=display, args=(env, path, visitedNodes, grid_res), daemon=True)
    display_thread.start()

    # Wait for camera to find Thymio and goal

    while True:
        if camera.Status == False:
            print("Waiting for camera to start...")
        else: 
            if Mes_car.found:
                print("Thymio found !")
            if Mes_goal.x != 0 or Mes_goal.y != 0:
                print("Goal found !")
            if Mes_car.found and (Mes_goal.x != 0 or Mes_goal.y != 0):
                print("Let's go !")
                break

            print("Waiting for camera to find Thymio and goal...")

        # Check if escape key pressed
        if keyboard.is_pressed('esc'):
            break
            
        time.sleep(0.1)

    # Init Kalman filter
    time.sleep(3)
    kalman = Kalman(Mes_car, TIMESTEP)

    # Init timer
    start = time.time()
    current = start

    print("Starting main loop...")

    print(f'Robot position: {(Mes_car.position.x, Mes_car.position.y)} and angle: {Mes_car.direction}')
    print(f'Goal position: {(Mes_goal.x, Mes_goal.y)}')
#####################################################################################################################
    # Init loop
    while True:
        # Update motion every 0.1 seconds
        deltaT = current - start
        if deltaT < TIMESTEP:
            time.sleep(0.01)
            current = time.time()
            continue

        # Update env with Kalman
        if not GOAL_REACHED:
            newcar, newspeed = kalman.kalman_filter(input, thymio.motors, Mes_car, deltaT)
            env.update(Environment(newcar, env.map, Mes_goal))
        
        # Compute path if needed
        if GLOBAL_PLANNING:
            thymio.set_variable(Motors(0,0))
            time.sleep(0.2)
            print("Planning...")

            visitedNodes.clear()
            path.clear()
            extended_obs.clear()

            calculate_path(env, path, extended_obs, visitedNodes, SIZE_THYM, False)
            if len(path) != 0:
                GLOBAL_PLANNING = False
                print("Planning finished !")
            
            
            # Update timer
            start = time.time() 
            current = start
            continue
        
        # Compute distance to next checkpoint and update accordingly 
        if not GOAL_REACHED and len(path) > 0:
            dist_to_checkpoint = env.robot.position.dist(path[0])
        else:
            dist_to_checkpoint = 0

        if dist_to_checkpoint >= LOST_TRESH:
            # If lost, recompute path on next iteration
            print("Lost !")
            GLOBAL_PLANNING = True
            continue

        # Check if goal reached
        if env.robot.position.dist(env.goal) <= REACH_GOAL_TRESH:
            print("Goal reached !")
            thymio.set_variable(Lights([0,255,0]))   # Light up the Thymio !
            thymio.set_variable(Motors(0,0))
            time.sleep(0.2)
            while True:
                print("Press 'esc' to end programm")
                if keyboard.is_pressed('esc'):
                    display_thread.join()
                    thymio_thread.join()
                    camera_thread.join()
                    thymio.stop()
                    exit()
                    
                time.sleep(0.3)
                
        if dist_to_checkpoint <= REACH_TRESH:
            # If sufficiently close to checkpoint, remove it from path and go to next one 
            print("Checkpoint reached !")
            if len(path) > 1:
                path.pop(0)  
                continue    
            else : 
                print("Path finished !")
                Global_planning = True
                continue

        ## Update motion
        if not GLOBAL_PLANNING:
            # PD controller
            current = time.time()
            dt = current - start
            motor_L, motor_R, theta_err = controller(env.robot, path[0], BASESPEED, theta_err, dt, GOAL_REACHED)

            # Obstacle avoidance
            prox_array = thymio.sensors.prox
            obstacle_detected, addLeft , addRight = obstacle_avoidance(prox_array)
            if obstacle_detected : 
                motor_L +=  addLeft
                motor_R +=  addRight
                position_temp_obstacle(prox_array, env.robot, temp_obstacles)
                if len(path) > 1 and GO_NEXT:           #update to the next checkpoint 
                    path.pop(0)
                    GO_NEXT = False
                    continue
            else:
                GO_NEXT = True   
                temp_obstacles.clear()             

            input = Motors(int(motor_L), int(motor_R))
            # input = Motors(0,0)
            thymio.set_variable(input)

        # Update timer
        start = time.time()
        current = start

        # Check if escape key pressed
        if keyboard.is_pressed('esc'):
            thymio.set_variable(Motors(0,0))
            break

#####################################################################################################################
    # Stop Thymio
    display_thread.join()
    thymio_thread.join()
    camera_thread.join()
    thymio.stop()
    exit()

