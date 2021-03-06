import numpy as np
import time

# Function to check if the Rover has been stuck in a postion for a 
# a certain amount of time
def is_rover_stuck(Rover):
    if Rover.vel <= 0.2 and Rover.vel >= -0.2 and (Rover.total_time - Rover.stuck_time) >= Rover.stuck_wait_time:
        return True

    return False

def rover_issue_correction(Rover):
    Rover.throttle = 0
    # Release the brake to allow turning
    Rover.brake = 0
    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
    Rover.steer = -15 # Could be more clever here about which way to turn

def rover_prepare_for_driving(Rover):
    # Set throttle back to stored value
    Rover.throttle = Rover.throttle_set
    # Release the brake
    Rover.brake = 0
    # Set steering to average angle clipped to the range +/- 15
    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
    Rover.mode = 'forward'

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward: 
                if is_rover_stuck(Rover):
                    # Remove throttle, stop steering and apply brake
                    Rover.throttle = 0
                    Rover.steer = 0
                    Rover.brake = Rover.brake_set
                    Rover.mode = 'stuck'
                    Rover.stuck_time = Rover.total_time
                    Rover.stuck_yaw_sample = Rover.yaw
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                elif Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
        ## Handle conditions when Rover is stuck/Jammed.
        elif Rover.mode == 'stuck':
            if abs(Rover.yaw - Rover.stuck_yaw_sample) >= Rover.stuck_yaw_tolerance:
                rover_prepare_for_driving(Rover)
                Rover.stuck_time = Rover.total_time
                Rover.stuck_yaw_sample = Rover.yaw
            # Now we're stopped and we have vision data to see if there's a path forward
            else:
                rover_issue_correction(Rover)

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    rover_issue_correction(Rover)
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    rover_prepare_for_driving(Rover)
            
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

