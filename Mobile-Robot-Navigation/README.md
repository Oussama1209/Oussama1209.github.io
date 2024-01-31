# BOMR_Project

# Course Project - Basics of Mobile Robotics

## Project Overview
The "Course Project - [ Basics of Mobile Robotics ](https://moodle.epfl.ch/course/view.php?id=15293) focuses on creating an environment where a Thymio robot navigates through obstacles using global and local navigation techniques. The project encompasses the following key aspects:

1. **Create an Environment**
   - The environment include a set of obstacles (black paper sheets).
   - The Thymio is expected to navigate and avoid flat obstacles without relying on sensors to detect them.

2. **Find the Best Path**
   - The Thymio's objective is to travel from its position in the map to a target.

3. **Motion Control & Pose Estimation**
   - Implement motion control to guide the robot along the path.
   - Achieve accurate pose estimation through Kalman filtering.

4. **Avoid Obstacles**
   - During navigation, the Thymio must employ local navigation techniques to avoid physical obstacles (3D white blocks).
   - Those obstacles are not detected in the vision 

## Installation
To install and set up the project, you can download it from our [GitHub repository](https://github.com/Thebestvic07/BOMR_Project.git)

## Usage
To run the project, execute the main.py script. More explantion in the report.

## Vision and Aruco Configuration

Captures video frames from the camera, processes ArUco markers, and displays the robot's position, orientation, and goal information. The Thymio navigates until the goal is reached or the user interrupts the program by pressing 'q'.

## Kalman Filter
The Kalman filter function implement an EKF that estimates the current state.
For this it uses the camera & motor speed measurement, the motor input, the previous state and the dynamical model. 

## Global Navigation
The visualization includes creating plots to display the environment, A* algorithm progress, and the final path. Obstacles are shown in black, free cells in white, the start position in green, the goal position in red, and the path in blue.

## Motion Control
The functions get the the angle, position of the Thymio and the next point in the global path. It will compute the velocity of both wheels depending on a PD controler.

## Local Navigation
The obstacle avoidance algorithm uses a potential field navigation approach. The obstacles are identified based on proximity sensor readings and calculates the velocity adjustments in the local frame to navigate around them.

## Report
Find a complet report in a Jupyter Notebook on our [GitHub repository](https://github.com/Thebestvic07/BOMR_Project.git). The report provides a detailed explanations of our code.

The project was made by : Beuret Sylvain, Labbe Victor, Jaffal Oussama & Wilhelm Laetitia
