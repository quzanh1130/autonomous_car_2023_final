
import cv2
import traceback
import time
from traffsign.traffic_sign_detection import SignDetector
from object.object_detection import ObjectDetector
# from lane.lane_line_detection import laneDetector
from lane.lane_line_segmentation import laneDetector
# from lane.lane_line_segmentation_v1 import laneDetector
from utils.carcontroler import CarController
from platform_modules.button_reader import ButtonReader
from platform_modules.car_guard import CarGuard
from platform_modules.camera import Camera
from platform_modules.lcd_display import LCDDisplay
from platform_modules.motor_controller import MotorController
import global_storage as gs
import config as cf
from utils.queue_handle import *


def main():
    # Init camera
    camera = Camera()
    camera.start()
   
    laneDetect = laneDetector()
    laneDetect.start()
    
    signDetect = SignDetector()
    signDetect.start()
    
    # objectDetect = ObjectDetector()
    # objectDetect.start()
    
    # # Init motor controller
    motor_controller = MotorController()
    motor_controller.start()
    
    # # Init button reader
    button_reader = ButtonReader()
    button_reader.start()
    
    # # Car guard
    # # Stop car when hitting obstacle or when user presses button 4
    guard = CarGuard()
    guard.start()
    
    lcd_display = LCDDisplay()
    lcd_display.start()
    
    carcontrol = CarController()
    
    # Variables for fps calculation
    start_time = time.time()
    frame_count = 0
    # Grab images and 

    while not gs.exit_signal:
        try:
            mask = gs.current_img
            rgb =  get_fast(gs.rgb_frames)
            # rgb = cv2.resize(rgb, (640, 480))
            
            if (gs.show_rgb):
                cv2.imshow("rgb", rgb)
                cv2.waitKey(1)
            
            # print("signs: ", gs.signs)
            # print("objects: ", gs.objects)
            # frame = cv2.hconcat([rgb, mask])
            if gs.show_mask:
                cv2.imshow('Result', mask)
                cv2.waitKey(1)
            frame_count += 1
            
            # Calculate fps every second
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Check if one second has passed
                fps = frame_count / elapsed_time
                gs.fps = fps
                print(f"FPS: {fps:.2f}")
                # Reset counters for the next second
                start_time = time.time()
                frame_count = 0
        
            throttle, steering_angle = carcontrol.decision_control(rgb, mask, gs.signs, gs.objects)
            # print("cc")
            
            if throttle != 0:
                gs.speed = 10
                cf.THROTTLE_NEUTRAL = 622
            else:
                gs.speed = 0
                cf.THROTTLE_NEUTRAL = 614
            gs.steer = steering_angle
            print("throtle1: " + str(gs.speed) + " steering1: " + str(gs.steer))
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error) # An exception occurred:
            traceback.print_exc()
            continue
        

        
    camera.join()
    signDetect.join()
    # objectDetect.join()
    laneDetect.join()
    motor_controller.join()
    guard.join()
    button_reader.join()
    lcd_display.join()
if __name__ == '__main__':
    main()
