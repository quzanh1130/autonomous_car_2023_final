import numpy as np
import cv2
import time
from utils.param import Param
import global_storage as gs
from utils.PID_Fuzzy import Fuzzy_PID, PID
from utils.queue_handle import *

class CarController():
    def __init__(self):
        self.rgb = 0
        self.image = 0
        
        self.roi1 = 0.85
        self.roi2 = 0.3
        
        self.param = Param()
        self.turning_time = 0
        self.seeSignTime = 0
        self.last_sign_time = 0
        self.lastSignDetection = ''
        self.lastSign = ''
        self.countObject = 0
        self.haveObject = 0
        
        self.countTurn1 = 0
        self.countTurn2 = 0
        self.turnStatus = 0
        
        self.throttle = self.param.maxThrotle
        self.steering_angle = self.param.steering
        self.im_height, self.im_width = 240, 320
        self.center = self.im_width // 2
        
        self.turnDirection = 0
        
        # self.pid_controller = Fuzzy_PID(15,0,1,0,1, 0)
        self.pid_controller = PID(1.5,0,0)
        # self.pid_controller.setKp(15, 0)
        # self.pid_controller .setKi(1, 0)
        # self.pid_controller.setKd(1, 0)
        self.pid_controller.setSampleTime(0.015) # Set the sample time (adjust as needed)
        setpoint = 0.0
        self.pid_controller.setSetPoint(setpoint)
    
    def birdview_transform(self, img):
        """
            Apply bird-view transform to the image
        """
        IMAGE_H = 240
        IMAGE_W = 320
 
        src = np.float32([[0, IMAGE_H], [320, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
        dst = np.float32([[120, IMAGE_H], [320 - 120, IMAGE_H], [-80, 0], [IMAGE_W+80, 0]])
        M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
        return warped_img
    
    def find_left_right_points(self):
        
        self.img_birdview = self.birdview_transform(self.image)
        
        if (gs.show_draw):
            self.rgb[:, :] = self.birdview_transform(self.rgb)
 
        
        # # ====================================================================
        interested_line_y = int(self.im_height * self.roi1)
        interested_line_y2 = int(self.im_height * self.roi2)
        interested_line_x = int(self.im_width * 0.5)
        
        if self.rgb is not None and gs.show_draw:
            cv2.line(self.rgb, (interested_line_x, 0),
                    (interested_line_x, self.im_height), (0, 0, 255), 2)
                    
            cv2.line(self.rgb, (0, interested_line_y),
                    (self.im_width, interested_line_y), (0, 0, 255), 2)
            cv2.line(self.rgb, (0, interested_line_y2),
                    (self.im_width, interested_line_y2), (0, 0, 255), 2)     
            
        interested_line = self.img_birdview[interested_line_y, :]
        interested_line2 = self.img_birdview[interested_line_y2, :]
        
        # interested_mid = self.img_birdview[:, interested_line_x]

        # # Detect left/right points
        self.left_point = -1
        self.right_point = -1
        
        self.left_point2 = -1
        self.right_point2 = -1
        

        self.haveLeft = 0
        self.haveRight = 0
        
        self.haveLeft2 = 0
        self.haveRight2 = 0

        # Define a helper function for finding the left and right points
        def find_point(interested_line):
            left_point, right_point = -1, -1

            # Search for left point
            for x in range(self.center, 0, -1):
                if interested_line[x] > 0:
                    left_point = x
                    break

            # Search for right point
            for x in range(self.center + 1, self.im_width):
                if interested_line[x] > 0:
                    right_point = x
                    break

            return left_point, right_point

        # Optimize the search for interested_line
        self.left_point, self.right_point = find_point(interested_line)
        self.haveLeft = 1 if self.left_point != -1 else 0
        self.haveRight = 1 if self.right_point != -1 else 0

        # Optimize the search for interested_line2
        self.left_point2, self.right_point2 = find_point(interested_line2)
        self.haveLeft2 = 1 if self.left_point2 != -1 else 0
        self.haveRight2 = 1 if self.right_point2 != -1 else 0
            
        if (self.haveLeft != 0 and self.haveRight !=0):
            self.len_line = 2
        elif (self.haveLeft == 0 and self.haveRight ==0):
            self.len_line = 0
        else:
            self.len_line = 1
        if gs.show_birdview:
            cv2.imshow("img_birdview", self.img_birdview)
            cv2.waitKey(1)
            
        # find_contours = None
        # find_contours, _ = cv2.findContours(self.img_birdview, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if find_contours != None:
        #     num_edges = len(find_contours)
        #     print("Countour dsdasdasdas: ", num_edges)
        # ============================================================================
        # print("midpoin: ", self.mid_point)
        # print("leftpoin 1: ", self.left_point)
        # print("rightpoint 1: ", self.right_point)
        # print("leftpoin 2: ", self.left_point2)
        # print("rightpoint 2: ", self.right_point)
        
        # if abs(self.left_point - self.right_point2) < 30:
        #     self.right_point = self.left_point
        #     self.left_point = -1
        # elif abs(self.right_point - self.left_point2) < 30:
        #     self.left_point = self.right_point 
        #     self.right_point = -1
              
        if self.left_point != -1 and self.right_point != -1:          
            # if (self.left_point<130) and (self.right_point<185):
                # self.left_point = self.right_point - 60
            # if (self.left_point>233) and (self.right_point>407):
            #     self.right_point = self.left_point + 166
            # if (self.left_point<233) and (self.right_point>407) and (abs(self.right_point-self.left_point)>166):
            #     self.left_point = 235
            #     self.right_point = 405
            if (abs(self.right_point-self.left_point)<30):
                print("chummm")
                self.left_point = 130
                self.right_point = 185

        # Predict right point when only see the left point
        if self.left_point != -1 and self.right_point == -1:
            if (self.left_point > 120 ) and (self.left_point < 160):
                print("one left -1 ")
                self.right_point = self.left_point + 130
            else:
                print("one left - 2")
                self.right_point = self.left_point + 130

        # Predict left point when only see the right point
        if self.right_point != -1 and self.left_point == -1:
            if (self.right_point >150) and (self.right_point < 230):
                self.left_point = self.right_point - 130
                print("one right")
            else:
                self.left_point = self.right_point - 130
        # =====================================================================================
        
        if self.rgb is not None and gs.show_draw:
            # if self.mid_point != -1:
            #     self.rgb = cv2.circle(
            #         self.rgb, (interested_line_x, self.mid_point), 5, (0, 0, 0), -1)
            if self.left_point != -1:
                self.rgb = cv2.circle(
                    self.rgb, (self.left_point, interested_line_y), 5, (255, 255, 0), -1)
            if self.right_point != -1:
                self.rgb = cv2.circle(
                    self.rgb, (self.right_point, interested_line_y), 5, (0, 255, 0), -1)
            if self.left_point2 != -1:
                self.rgb = cv2.circle(
                    self.rgb, (self.left_point2, interested_line_y2), 5, (255, 255, 0), -1)
            if self.right_point2 != -1:
                self.rgb = cv2.circle(
                    self.rgb, (self.right_point2, interested_line_y2), 5, (0, 255, 0), -1)
        if gs.show_draw:
            cv2.imshow('Result', self.rgb)
            cv2.waitKey(1)
        
        
    def calculate_control_signal(self, objects):
        
        # Find left/right points  
        self.find_left_right_points()

        # Calculate deviation from the center of the lane
        # self.center = self.image.shape[1] // 2
         ###########################################objects detection##########################################  
        # if (self.right_point2 != -1 and (self.right_point2 > 150 and self.right_point2 < 170)):
        #     print("Vat Lefttttttttttttttttttttt")
        #     self.haveObject = -1
        #     self.countObject += 1
        
        # if (self.left_point2 != -1 and (self.left_point2 > 130 and self.left_point2 < 150)):
        #     print("Vat Right888888")
        #     self.haveObject = 1
        #     self.countObject += 1
           
        # objects_value = objects[:]
        # if (self.left_point2!= -1 and self.right_point2 != -1) and (abs(self.left_point- self.right_point) > 80):
        #     print("Trc vat trai phai")
        #     print("leftpoin 1: ", self.left_point)
        #     print("rightpoint 1: ", self.right_point)
        #     print("leftpoin 2: ", self.left_point2)
        #     print("rightpoint 2: ", self.right_point2)
        #     print("-------------haveObject:", self.haveObject)
        #     if (self.left_point2 >= 60 and self.left_point2 <= 100) and (self.right_point2 >= 150 and self.right_point2 <= 210):
        #         print("Vat Phaiiiiiiiiiiii")
        #         self.haveObject = 1
        #         self.countObject += 1 
        #     elif (self.left_point2 >= 100 and self.left_point2 <= 149) and (self.right_point2 >= 190 and self.right_point2 <= 260):
        #         print("Vat Traiiiiiiiiiiiiiii")
        #         self.haveObject = -1
        #         self.countObject += 1
          #####################################################################################              
            
        if self.left_point != -1 and self.right_point != -1:
             ##############################objects detection##########################################################  
            # if self.countObject >= self.param.maxCountObject and self.haveObject == -1:
            #     # mid_left_point = (middle_point + self.left_point) // 2
            #     # x_offset = self.center - mid_left_point 
            #     self.right_point += 30
            #     print("Object on the left road")
                 
            # elif self.countObject >= self.param.maxCountObject and self.haveObject == 1:
            #     # mid_left_point = (middle_point + self.left_point) // 2
            #     # x_offset = self.center - mid_left_point   
            #     self.left_point -= 30    
            #     print("Object on the right road")
        #####################################################################################       
            if self.left_point < 120:      
                # print("333333333333333333333333333333333")
                self.right_point += 15  
            middle_point = (self.right_point + self.left_point) // 2
            
            x_offset = self.center - middle_point
            
            self.pid_controller.update(x_offset)
            
            # Get the calculated steering angle from the PID controller
            steering_angle_fuzzy = self.pid_controller.output
            
            # Normalize the steering angle to the range -1 to 1
            steering_angle_normalized = -float(steering_angle_fuzzy) 
            steering_angle_normalized = max(-60, min(60, steering_angle_normalized))
            print("x-offset: ", x_offset)
            if (gs.emergency_stop == True):
                # print("******** clear PID")
                self.pid_controller.clear_stop()
        else:
            steering_angle_normalized = 0

        self.steering_angle = steering_angle_normalized
        
        
    def decision_control(self, rgb, mask, signs, objects):
        self.rgb = rgb
        self.image = mask
        self.calculate_control_signal(objects)
        signs_value = signs[:]
        if signs_value != [] and self.lastSignDetection == '':
            for sign in signs:
                class_name = sign
                if class_name == 'left':
                    self.lastSignDetection = 'left'
                    self.seeSignTime = time.time()
                elif class_name == 'right':
                    self.lastSignDetection = 'right'
                    self.seeSignTime = time.time()
                elif class_name == 'stop' and self.lastSign != 'stop':
                    self.lastSignDetection = 'stop'
                    self.seeSignTime = time.time()
                
        print("LastSignDetection:",self.lastSignDetection)

        # if see the sign will decreace the throttle
        # if self.steering_angle != 0 and self.lastSignDetection != '' and self.turning_time == 0:  
        #     self.throttle = self.param.minThrottle 
           
        # will go ahead when see the sign when it don't see one or two line
        # if self.lastSignDetection != '' and self.turning_time == 0 and (self.haveLeft == 0 or self.haveRight ==0) and signs_value ==[]:
        #     self.steering_angle = self.param.steering # 0
        # self.lastSignDetection = 'right'

         # set turning time for right -------- Turn 90*    
        # if (self.haveLeft == 0 and self.haveRight == 0  and self.haveLeft2 == 0 and self.haveRight2 == 0 and self.turning_time == 0 and self.lastSignDetection == ''):
        #     self.countTurn1 += 1
        #     self.countTurn2 += 1
        #     if (self.haveLeft == 1 and self.haveRight == 0  and self.haveLeft2 == 0 and self.haveRight2 == 0 and self.turning_time == 0 and self.lastSignDetection == ''):
        #         self.countTurn1 += 1
        #         self.countTurn2 = 0
        #         print("++++++++++++++=Count1: ", self.countTurn1)
        #     elif (self.haveLeft == 0 and self.haveRight == 1  and self.haveLeft2 == 0 and self.haveRight2 == 0 and self.turning_time == 0 and self.lastSignDetection == ''):
        #         self.countTurn1 = 0
        #         self.countTurn2 += 1
        #         print("++++++++++++++=Count2: ", self.countTurn2)
        # else:
        #     # print("++++++++++1++++=Count2: ", self.countTurn2)
        #     self.countTurn1 = 0
        #     self.countTurn2 = 0
        

        
        # ==============================RIGHT============================
        # set turning time for right -------- traffsign
        if  (self.turning_time == 0 and self.lastSignDetection == 'right' and signs_value ==[] and self.haveRight ==0):
            self.turning_time = self.param.maxTurnTimeSign
            self.last_sign_time = time.time()
            self.turnStatus = 1
            print("******************turn right 1 ")
        
        # # set turning time for left -------- 90*
        # if (self.haveLeft == 0 and self.haveRight == 0  and self.haveLeft2 == 0 and self.haveRight2 == 0 and self.turning_time == 0 and self.lastSignDetection == '')\
        # or (self.haveLeft == 1 and self.haveRight == 0  and self.haveLeft2 == 0 and self.haveRight2 == 0 and self.turning_time == 0 and self.lastSignDetection == ''):
        # or (self.haveLeft == 1 and self.haveRight == 0 and self.left_point2 >= 150 and self.left_point2 <= 170 )\
        # or (self.haveLeft == 0 and self.haveRight == 1 and self.right_point2 >= 150 and self.right_point2 <= 170 ):

        if (self.haveLeft == 1 and self.haveRight == 0  and self.turning_time == 0 and self.lastSignDetection == ''):
            self.countTurn1 += 1
            print("++++++++++++++=Count1", self.countTurn1)
        else:
            self.countTurn1 = 0
        
        
        if (self.turnDirection == 1 and self.countTurn1 >= 3 and self.turning_time == 0 and self.lastSignDetection == ''):
            self.turning_time = self.param.maxTurnTime90
            self.lastSignDetection = 'right'
            self.last_sign_time = time.time()
            self.turnStatus = 2
            print("*******************turn right 2")
        
        # set turning time for right -------- Vong Xuyen
        if (gs.haveObject == True and self.haveRight == 0 and self.turning_time == 0 and self.lastSignDetection == ''):
            self.turning_time = self.param.maxTurnTimeXuyen
            self.lastSignDetection = 'right'
            self.last_sign_time = time.time()
            self.turnStatus = 3
            print("****************turn right 3")
        
        # ============================= LEFT =====================================
        # set turning time for left -------- traffsign
        if  (self.turning_time == 0 and self.lastSignDetection == 'left' and signs_value ==[] and self.haveLeft ==0):
            self.turning_time = self.param.maxTurnTimeSign
            self.last_sign_time = time.time()
            self.turnStatus = 1
            print("******************turn left 1 ")
        
        # # set turning time for left -------- 90*
        # if (self.haveLeft == 0 and self.haveRight == 0  and self.haveLeft2 == 0 and self.haveRight2 == 0 and self.turning_time == 0 and self.lastSignDetection == '')\
        # or (self.haveLeft == 0 and self.haveRight == 1  and self.haveLeft2 == 0 and self.haveRight2 == 0 and self.turning_time == 0 and self.lastSignDetection == ''):
        # or (self.haveLeft == 1 and self.haveRight == 0 and self.left_point2 >= 150 and self.left_point2 <= 170 )\
        # or (self.haveLeft == 0 and self.haveRight == 1 and self.right_point2 >= 150 and self.right_point2 <= 170 ):
        if (self.haveLeft == 0 and self.haveRight == 1  and self.turning_time == 0 and self.lastSignDetection == ''):
            self.countTurn2 += 1
            print("++++++++++++++=Count2", self.countTurn2)
        else:
            self.countTurn2 = 0
        
        if (self.turnDirection == 2 and self.countTurn2 >= 3 and self.turning_time == 0 and self.lastSignDetection == ''):
            self.turning_time = self.param.maxTurnTime90
            self.lastSignDetection = 'left'
            self.last_sign_time = time.time()
            self.turnStatus = 2
            print("*******************turn left 2")
        
        # set turning time for left -------- Vong Xuyen
        if (gs.haveObject == True and self.haveRight == 0 and self.turning_time == 0 and self.lastSignDetection == ''):
            self.turning_time = self.param.maxTurnTimeXuyen
            self.lastSignDetection = 'left'
            self.last_sign_time = time.time()
            self.turnStatus = 3
            print("****************turn left 3")
        
        # =======================STOP======================== 

        # set turning time for stop sign imediately
        if self.lastSignDetection == 'stop' and self.turning_time == 0:
            self.turning_time = self.param.stoptime
            self.last_sign_time = time.time()
            print("stop")
            self.turnStatus = 1
        
        #go straight when don't have two lane line
        if self.haveLeft == 0 and self.haveRight == 0 and self.turnStatus == 0:
            self.steering_angle = 0
        
                
        if (self.turnDirection == 3 ):
            self.steering_angle = 20
            self.turnDirection = 0
            print("*******************turn to 3 line")
        
        # if have turnning time
        if (time.time() - self.last_sign_time) >= 0 and (time.time() - self.last_sign_time) <= self.turning_time and self.lastSignDetection != '':         
            if self.lastSignDetection != '' and self.turnStatus == 1:
                if self.lastSignDetection == 'left':
                    self.steering_angle = -50
                elif self.lastSignDetection == 'right':
                    self.steering_angle = 50
                elif self.lastSignDetection == 'stop':
                    self.throttle = 0
                    self.steering_angle = 0
            elif self.lastSignDetection != '' and self.turnStatus == 2:
                if self.lastSignDetection == 'left':
                    self.countTurn1 = 0
                    self.steering_angle = -60
                elif self.lastSignDetection == 'right':
                    self.countTurn2 = 0
                    self.steering_angle = 60
            elif self.lastSignDetection != '' and self.turnStatus == 3:
                if self.lastSignDetection == 'left':
                    self.steering_angle = -50
                elif self.lastSignDetection == 'right':
                    self.steering_angle = 50
            # clear all when have two line
            if (self.len_line == 2) and (time.time() - self.last_sign_time) >= self.param.minTurnTime90:
                self.turning_time = 0
                self.lastSign = self.lastSignDetection
                self.lastSignDetection = ''
                self.seeSignTime = 0
                self.last_sign_time = 0
                self.countObject = 0
                self.haveObject = 0
                self.turnStatus = 0
                self.countTurn1 = 0
                self.countTurn2 = 0
                print("clear all 1")
        # clear when out of time
        elif ((time.time() - self.last_sign_time) < 0 or (time.time() - self.last_sign_time) >= self.turning_time) and self.lastSignDetection != '' and self.turning_time != 0 and self.len_line == 2 :
            self.turning_time = 0
            self.lastSign = self.lastSignDetection
            self.lastSignDetection = ''
            self.seeSignTime = 0
            self.last_sign_time = 0
            self.countObject = 0
            self.haveObject = 0
            self.turnStatus = 0
            self.countTurn1 = 0
            self.countTurn2 = 0
            print("clear all 2")
        
        # # Reset when see sign but don't turn
        if (self.lastSignDetection != '' and self.turning_time == 0 and (time.time() - self.seeSignTime) >= 4) or gs.emergency_stop == True:
            self.turning_time = 0
            self.lastSign = self.lastSignDetection
            self.lastSignDetection = ''
            self.last_sign_time = 0
            self.seeSignTime = 0
            self.countObject = 0
            self.haveObject = 0
            self.turnStatus = 0
            self.countTurn1 = 0
            self.countTurn2 = 0
            print("New round")    
            
        return self.throttle, self.steering_angle
    
    
