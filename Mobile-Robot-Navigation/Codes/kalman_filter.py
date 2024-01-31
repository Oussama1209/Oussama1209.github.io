from .utils.data import *
import numpy as np
from scipy.signal import butter, lfilter

class Kalman:
    # Constants
    THYMIO_WIDTH = 2.5
    TIMESTEP = 0.1
    SPEEDCONV = 0.01 # 1 motor unit = 0.1 case/s

    MOT_VAR = 6.25   # +- 5      --> sigma = 2.5 --> var = 6.25
    POS_VAR = 0.125  # +- 0.5 case --> sigma = 0.25 --> var = 0.125
    DIR_VAR = 0.01   # +- 5°     --> sigma = 2.5° --> sigma = 0.1 rad --> var = 0.01

    def __init__(self, initial_state: Robot, timestep = TIMESTEP) -> None:
        # Initialize memory variables for kalman filter
        self.robot = initial_state
        self.speed = [0.0,0.0]
        self.input = Motors(0,0)
        self.cov  = np.diag([0,0,Kalman.POS_VAR,Kalman.POS_VAR,Kalman.DIR_VAR])
        self.dt = timestep

        # covariance matrices
        self.R = np.diag([Kalman.MOT_VAR, Kalman.MOT_VAR, Kalman.POS_VAR, Kalman.POS_VAR, Kalman.DIR_VAR])
        self.Q = np.array([ [1, 0 , self.dt, self.dt, 0],    
                            [0, 2/Kalman.THYMIO_WIDTH, 0, 0, self.dt],
                            [self.dt, 0, 1, 0, 0],
                            [self.dt, 0, 0, 1, 0],
                            [0, self.dt, 0, 0, 1]]
                        ) 


    def kalman_filter(self, mot_input : Motors, mot_mes : Motors, cam_mes : Robot, dt = None):
        """
        This function implement an EKF that estimates the current state 
        For this it uses the camera & motor speed measurement, the motor input and the previous state
        
        param mot_input        : input motor speed (Motor object)    
        param mot_mes          : measured motor speed (Motor object)
        param rob_mes          : measured position via camera (Robot object)
        param dt               : timestep (default = TIMESTEP of Kalman class)
        
        return mot_est         : new a posteriori motor speed estimation
        return pos_est         : new a posteriori position estimation 
        """
        ## Transposing arguments into arrays for easier manipulation
        state_prev = np.array([self.speed[0], self.speed[1], self.robot.position.x, self.robot.position.y, self.robot.direction])
        input = np.array([mot_input.left - self.input.left, mot_input.right - self.input.right])
        
        dt = self.dt if dt == None else dt

        ## Prediciton through the a priori estimate
        a_priori_state, G = motion_model(state_prev, input, dt)    
        a_priori_cov      = G @ self.cov @ G.T + self.Q

        ## Measurement model
        state_meas, C, R = measurement_model(mot_mes, cam_mes, self.R)

        ## Gain and innovation computation
        K = a_priori_cov @ C.T @ np.linalg.inv(C @ a_priori_cov @ C.T + R)  
        innovation = state_meas - C @ a_priori_state

        ## Estimation with a priori and innovation
        a_posteriori_state = a_priori_state + K @ innovation
        a_posteriori_cov = (np.eye(5) - K @ C) @ a_priori_cov
        
        ## Update the state
        speed_est = a_posteriori_state[0:2]
        pos_est = a_posteriori_state[2:5]
        cov_est = a_posteriori_cov

        self.speed = speed_est
        self.input = mot_input
        self.robot = Robot(Point(pos_est[0], pos_est[1]), pos_est[2])
        self.cov = cov_est
  
        return self.robot, self.speed


## Our motion model is non linear.
# It is ruled by the following laws :

    #   Fw speed  v      : v+ = v + K(dmot_l + dmot_r) / 2             (K = speed/motor units conversion factor) 
    #   Rot speed w      : w+ = w + K(dmot_l - dmot_r) / THYMIO_WIDTH  (dmot = new_mot_input - prev_mot_input)
    #   Position X       : x+ = x + v*cos(theta)*dt
    #   Position Y       : y+ = y + v*sin(theta)*dt
    #   Direction theta  : theta+ = theta + w*dt

# Therefore we'll take our state X = [v, w  x, y, theta]
# Our input is U = [dmot_l, dmot_r] (the delta between the new and the previous input motor speed) 
# Our measurement Y = [S_mot_l, S_mot_r, x_mes, y_mes, theta_mes] (with S for sensor)
# The motors' speed are measured at all time while the position are only measured while the camera is active

def motion_model(prev_state, input, dt):
    """
    This function returns the state at the next time step and its Jacobian 
        param prev_state : previous state as an array(5,1)
        param input      : new input as an array(2,1)
        param dt         : timestep

        return estimated state as an array(5,1)
        return Jacobian as a numpy matrix (5,7)
    """
    ## Unpacking the previous state and the input
    v = prev_state[0] 
    w = prev_state[1]
    x = prev_state[2]
    y = prev_state[3]
    theta = prev_state[4]
    dmot_l = input[0]
    dmot_r = input[1]

    ## Estimating the new state
    est_state = np.array([0.0]*5)

    est_state[0] = v +   Kalman.SPEEDCONV * (dmot_l + dmot_r) / 2
    est_state[1] = w + Kalman.SPEEDCONV * (dmot_l - dmot_r) / Kalman.THYMIO_WIDTH 
    est_state[2] = x + v * np.cos(theta) * dt
    est_state[3] = y + v * np.sin(theta) * dt
    est_state[4] = angle_correction(theta + w * dt)

    ## Computing the Jacobian
    Jacobian = np.array(
        [
            [1.0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0],
            [np.cos(theta)*dt, 0, 1.0, 0, -v*np.sin(theta)*dt],
            [np.sin(theta)*dt, 0, 0, 1.0,  v*np.cos(theta)*dt],
            [0, dt, 0, 0, 1.0],
        ]
    )  

    return est_state, Jacobian


def measurement_model(mot_mes, rob_mes, R):
    """
    This function returns the measured state and the measurement model and covariance  
        param mot_mes : measured motor speed (Motor object)
        param rob_mes : measured position via camera (Robot object)
        param R       : default covariance matrix of the measurement

        return state_meas : measured state as an array
        return C          : measurement model s.t. y = C @ x 
        return R          : covariance matrix of the measurement
    """
    ## Check if we have a camera measurement
    if rob_mes.found == False:     # Pas de caméra -->  y = [S_mot_l, S_mot_r] = C @ [v, w, x, y, theta]
        print("No camera !")

        state_meas  = np.array([mot_mes.left, mot_mes.right])
        C = np.zeros((2,5))
        C[0:2,0:2] = [[1/Kalman.SPEEDCONV, -1/Kalman.SPEEDCONV],
                      [1/Kalman.SPEEDCONV,  1/Kalman.SPEEDCONV]]
        
        R = np.diag(R[0:2,0:2])
        
    else:                         # Pas de caméra -->   y = [S_mot_l, S_mot_r, x_mes, y_mes, theta_mes] = C @ [v, w, x, y, theta]
        state_meas  = np.array([mot_mes.left, mot_mes.right, rob_mes.position.x, rob_mes.position.y, angle_correction(rob_mes.direction)])

        C = np.zeros((5,5))
        C[0:2,0:2] = [[1/Kalman.SPEEDCONV, -1/Kalman.SPEEDCONV],
                      [1/Kalman.SPEEDCONV,  1/Kalman.SPEEDCONV]]
        C[2,2] = 1.0
        C[3,3] = 1.0
        C[4,4] = 1.0
        
        R = R

    return state_meas, C, R


def angle_correction(angle):
    """
    This function corrects the angle to be between -pi and pi
        param angle : angle to correct

        return corrected angle
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

