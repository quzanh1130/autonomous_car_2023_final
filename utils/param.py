import cv2
import onnxruntime
import numpy as np

class Param:
     def __init__(self):
        self.minThrottle = 0.25
        self.maxThrotle = 0.35
        self.steering = 0
        
        self.minTurnTimeSign = 2
        self.maxTurnTimeSign = 4
        
        self.minTurnTime90 = 1
        self.maxTurnTime90 = 3
        
        self.minTurnTimeXuyen = 3
        self.maxTurnTimeXuyen = 4
        
        self.maxCountObject = 3
        self.stoptime = 4
        # Initalize traffic sign classifier
        self.traffic_sign_model = cv2.dnn.readNetFromONNX("models/traffic_sign_classifier_2.onnx")
        # Load HaarCascade
        self.cascade = cv2.CascadeClassifier('object/car.xml')
        # Filter for object classifier to fill
        self.onnx_session = onnxruntime.InferenceSession('models/classification_model_v3.onnx')
        
        self.lane_segment_model_onnx = onnxruntime.InferenceSession('models/enet_v2.onnx')
    
