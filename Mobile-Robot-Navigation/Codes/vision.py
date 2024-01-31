import numpy as np
import cv2
import math
from .utils.data import *

############################################### SET WHAT IS BLACK AND WHAT IS WHITE  ###############################################
def is_black_cell(image):
    # Know if the grid cell contains more than 1/8 pixels who are black
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)

    # Calculate the percentage of black pixels in the image
    total_pixels = image.size
    black_pixels = total_pixels - cv2.countNonZero(binary_mask)
    percentage_black = (black_pixels / total_pixels) * 100

    # Set the threshold percentage
    threshold_percentage = 10  # Adjust this threshold as needed

    # Check if the percentage of black pixels exceeds the threshold
    if percentage_black > threshold_percentage:
        return True
    else:
        return False
    
# Change the image according to the percentage of black pixels
def change_cell(image):
    
    if is_black_cell(image):
        # Change the image to a black cell
        return cv2.imread('black_image.png')

    else:
        # Change the image to a white cell
        return cv2.imread('white_image.png')

######################################################### DISPLAY HELPERS   ###########################################################

def show_grid(image, grid_resolution):
    # Get image dimensions
    height, width = image.shape[:2]

    # Draw vertical grid lines
    for x in range(0, width, grid_resolution):
        cv2.line(image, (x, 0), (x, height), (0, 100, 0), 1)

    # Draw horizontal grid lines
    for y in range(0, height, grid_resolution):
        cv2.line(image, (0, y), (width, y), (0, 100, 0), 1)
    
    return image


def show_robot(frame, arucos, grid_resolution):
    robot_pos = robot_center_is(arucos,grid_resolution)
    # robot_pos = center_in_grid(arucos, 95, grid_resolution)
    angle = get_angle_of_robot(arucos)
    frame = draw_arrow(frame, arucos, angle)

    return frame, arucos, robot_pos, angle


def draw_arrow(image, arucos, angle, length=80, color=(0, 255, 0), thickness=3):

    if len(arucos) != 0:
        for i in range(len(arucos)):
            if arucos[i][2] == 95:
                arrow_start = (arucos[i][0], arucos[i][1])
                x =  int(round(arrow_start[0] + length * math.cos(angle)))
                y =  int(round(arrow_start[1] - length * math.sin(angle)))
                arrow_end = (x, y)
                 # Use cv2.arrowedLine to draw the arrow on the image
                image = cv2.arrowedLine(image, arrow_start, arrow_end, color, thickness)
                break
    return image


def draw_arrow_from_robot(image, robot, grid_res, length=80, color=(0, 255, 0), thickness=3):
    # if robot at (0,0) then no arrow
    if robot.position.x + robot.position.y  == 0:
        return image

    arrow_start = (robot.position.x + 1/2, robot.position.y + 1/2)
    arrow_start = (int(arrow_start[0]*grid_res), int(arrow_start[1]*grid_res))
    angle = robot.direction

    x =  int(round(arrow_start[0] + length * math.cos(angle)))
    y =  int(round(arrow_start[1] - length * math.sin(angle)))
    arrow_end = (x, y)

    # Use cv2.arrowedLine to draw the arrow on the image
    image = cv2.arrowedLine(image, arrow_start, arrow_end, color, thickness)

    return image


def draw_goal(image, arucos, grid_res, radius=10, color=(0, 0, 255), thickness=-1):
    position = get_goal_pos_in_pixel(arucos)

    if position == (0, 0):
        return image

    display_pos = (position[0] + grid_res/2, position[1] + grid_res/2)
    display_pos = (int(position[0]), int(position[1]))

    cv2.circle(image, display_pos, radius, color, thickness)
    return image


def draw_circle(image, grid_pos, grid_res, radius=5, color=(255, 0, 0), thickness=-1):
    grid_pos

    if grid_pos == (0, 0):
        return image

    display_pos = (grid_pos[0] + 1/2, grid_pos[1] + 1/2)
    display_pos = (int(display_pos[0]*grid_res), int(display_pos[1]*grid_res))

    cv2.circle(image, display_pos, radius, color, thickness)
    return image


def robot_center_is(arucos, grid_resolution):
    if len(arucos) !=0:
        for i in range(len(arucos)):
            if arucos[i][2] == 95:
                return (arucos[i][0] / grid_resolution, arucos[i][1] / grid_resolution)
    return (0, 0)
    

def center_in_grid(arucos, id, grid_resolution):
    if len(arucos) !=0:
        for i in range(len(arucos)):
            if arucos[i][2] == id:
                x = int(arucos[i][0]/grid_resolution)
                y = int(arucos[i][1]/grid_resolution)
                return (x, y)
    return (0, 0)


def get_goal_pos_in_pixel(arucos):
    if len(arucos) !=0:
        for i in range(len(arucos)):
            if arucos[i][2] == 99:
                x = int(arucos[i][0])
                y = int(arucos[i][1])
                return (x, y)
            
    return (0, 0)

######################################################### RECOGNITION FUNCTIONS   ###########################################################  

# Set aruco detector
def set_aruco():
    # Set the dictionary of aruco markers to 4x4_100
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    # Set the parameters for the detector
    parameters =  cv2.aruco.DetectorParameters()
    # Create the detector
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector

# Get information about the arucos
def get_info_arucos(corners, ids, image):

    arucos = []
    if len(corners) > 0:
 
        ids = ids.flatten()
 
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # Get the sides of the aruco
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Get the center of the aruco
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            
            # Store the information about the aruco
            arucos.append((cX, cY, markerID, (bottomLeft, topLeft)))
            
    return image, arucos

# Check if the goal is reached
def check_if_goal_reached(arucos, robot_pos, goal_pos):
    if len(arucos) !=0:
        if (robot_pos == goal_pos) and (robot_pos != (0, 0)) and (goal_pos != (0, 0)):
            return True
        else:
            return False
    else:
        return False

# Compute the angle of the robot
def get_angle_of_robot(arucos):
    if len(arucos) !=0:
        for i in range(len(arucos)):
            if arucos[i][2] == 95:
                x1, y1 = arucos[i][3][0][0], arucos[i][3][0][1]
                x2, y2 = arucos[i][3][1][0], arucos[i][3][1][1]
                angle = math.atan2(-(y2 - y1), x2 - x1)
                return angle
    return 0

def get_arucos(frame):
    detector = set_aruco()
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
    frame, arucos = get_info_arucos(markerCorners, markerIds, frame)
    return frame, arucos

def get_corners(arucos):
    pos1, pos2, pos3, pos4 = (0, 0), (0, 0), (0, 0), (0, 0)
    if len(arucos) !=0:
        for i in range(len(arucos)):
            if arucos[i][2] == 0:
                pos1 = (arucos[i][0], arucos[i][1])
            if arucos[i][2] == 1:
                pos2 = (arucos[i][0], arucos[i][1])
            if arucos[i][2] == 2:
                pos3 = (arucos[i][0], arucos[i][1])
            if arucos[i][2] == 3:
                pos4 = (arucos[i][0], arucos[i][1])
        return pos1, pos2, pos3, pos4       
    else:
        return (0, 0), (0, 0), (0, 0), (0, 0)
    
def get_dist_grid(arucos, default=25):
    if len(arucos) !=0:
        for i in range(len(arucos)):
            if arucos[i][2] == 0:
                x1, y1 = arucos[i][3][0][0], arucos[i][3][0][1]
                x2, y2 = arucos[i][3][1][0], arucos[i][3][1][1]
                dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                return int(dist)
    return default
    
def get_dist_height_circuit(arucos):
    if len(arucos) !=0:
        for i in range(len(arucos)):
            if arucos[i][2] == 0:
                for j in range(len(arucos)):
                    if arucos[j][2] == 3:
                        x1, y1 = arucos[i][0], arucos[i][1]
                        x2, y2 = arucos[j][0], arucos[j][1]
                        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                        return dist
    else:
        return 0
    
def get_dist_width_circuit(arucos):
    if len(arucos) !=0:
        for i in range(len(arucos)):
            if arucos[i][2] == 0:
                for j in range(len(arucos)):
                    if arucos[j][2] == 1:
                        x1, y1 = arucos[i][0], arucos[i][1]
                        x2, y2 = arucos[j][0], arucos[j][1]
                        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                        return dist
    else:
        return 0

######################################################### BUILDER FUNCTIONS   ###########################################################  
  
# Build a map from an image
def apply_grid(image, grid_resolution):

    # Get image dimensions
    height, width = image.shape[:2]

    # Create black and white images
    create_black_image(grid_resolution)
    create_white_image(grid_resolution)

    # Get resolution of the grid
    x_cells = int(width / grid_resolution)
    y_cells = int(height / grid_resolution)

    # Creating a 2D list of grid cells
    map = [[0 for _ in range(y_cells)] for _ in range(x_cells)]
    obstacles = []
    internal_map = [[0 for _ in range(y_cells)] for _ in range(x_cells)]

    # Initializing variables
    obstacles = []
    new_image = list(range(y_cells))
    final_image = list(range(x_cells))

    for y in range(y_cells):
        for x in range(x_cells):
            # Define the bounding box for the current grid cell
            x_min, y_min = x*grid_resolution, y*grid_resolution
            x_max, y_max = x_min + grid_resolution, y_min + grid_resolution

            # Crop the image to the bounding box
            cell_content = image[y_min:y_max, x_min:x_max]

            # Check if the cell is an obstacle
            if is_black_cell(cell_content):
                map[x][y] = 0
                obstacles.append(Point(x,y))
            else:
                map[x][y] = 1

            # Store the cell content in the grid
            internal_map[x][y] = change_cell(cell_content)

            # Add the cell in the column
            if x==0:
                new_image[y] = internal_map[x][y]
            else:
                new_image[y] = np.hstack((new_image[y], internal_map[x][y]))
        
        # Add the column in the image
        if y==0:
            final_image = new_image[y]

        if y!=0:
            final_image = np.vstack((final_image,new_image[y]))

    # Create the map object
    map = Map([Point(0,0), Point(width,0), Point(width, height), Point(0,height)], obstacles)
    return final_image, map

 
def activate_camera():
    # Open the camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        frame = cap.read()[1]

        cv2.imshow("Videa Stream", frame)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()

    
def set_param_frame(arucos):
    dist_aruco = get_dist_grid(arucos)
    width, height = 0, 0
    if dist_aruco != 0:
        width = int(get_dist_width_circuit(arucos)/dist_aruco) + 1
        height = int(get_dist_height_circuit(arucos)/dist_aruco) + 1
        width, height = width*dist_aruco, height*dist_aruco
    return width, height
    
def projected_image(frame, arucos, width, height):
    pos1, pos2, pos3, pos4 = get_corners(arucos)
    condition = width != 0 and height != 0 and pos1 != (0,0) and pos2 != (0,0) and pos3 != (0,0) and pos4 != (0,0)
    if condition:
        pts1 = np.float32([pos1, pos2, pos3, pos4])
        pts2 = np.float32([[0,0],[width,0],[width,height],[0,height]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        result = cv2.warpPerspective(frame,matrix,(width,height))
        return result
    else:
        return frame

def apply_grid_to_camera(default_grid_res=25):

    # Open the camera
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    print("Calibrating Map...")

    while cap.isOpened():
        # Capture a frame
        ret, frame = cap.read()

        # Check if the frame was successfully captured
        if ret:
            frame, arucos = get_arucos(cap.read()[1])
            grid_resolution = get_dist_grid(arucos, default_grid_res)

            frame, map = apply_grid(frame, grid_resolution)
            # Save the captured frame as an image
            cv2.imwrite("captured_frame.png", frame)
            print("Map built with resolution: ", grid_resolution)


            return map, grid_resolution
        else:
            print("Ret is False, trying again...")

    # Release the camera
    cap.release()


def create_white_image(grid_resolution):
    # Create a white image
    white_image = 255 * np.ones((grid_resolution, grid_resolution, 3), dtype=np.uint8)

    # Save the white image
    cv2.imwrite('white_image.png', white_image)

def create_black_image(grid_resolution):
    # Create a black image
    black_image = np.zeros((grid_resolution, grid_resolution, 3), dtype=np.uint8)

    # Save the black image
    cv2.imwrite('black_image.png', black_image)

def replace_white():
    return cv2.imread('white_image.png')

def replace_black():
    return cv2.imread('black_image.png')

    
######################################################### MAIN FOR CALIBRATION & TESTING   ###########################################################  

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    last_known_goal_pos = (0, 0)
    robot_pos = (0, 0)

    arucos = get_arucos(cap.read()[1])[1]

    grid_resolution = get_dist_grid(arucos)

    width, height = 707, 10007
    # width, height = set_param_frame(arucos)

    while cap.isOpened():
        frame = cap.read()[1]

        frame, arucos = get_arucos(frame)

        frame, arucos, robot_pos, angle = show_robot(frame, arucos, grid_resolution)
        goal_pos = center_in_grid(arucos, 99, grid_resolution)

        if goal_pos != (0, 0):
            last_known_goal_pos = goal_pos

        draw_goal(frame, arucos, grid_resolution)

        if check_if_goal_reached(arucos, robot_pos, last_known_goal_pos):
            print('Goal reached')
            break

        # frame = projected_image(frame, arucos, width, height)
        cv2.imshow("Video Stream", frame)

        print("Robot position: ", robot_pos)
        print("Goal position: ", last_known_goal_pos)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


