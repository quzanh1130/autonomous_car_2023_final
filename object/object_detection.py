import cv2
import numpy as np
import threading
import time
import global_storage as gs
from utils.param import Param

class ObjectDetector(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        # List of class names
        self.param = Param()
        self.classes = ["nocar","car" ]
        self.cascade = self.param.cascade
        self.onnx_session = self.param.onnx_session
        
    def run(self):
        while not gs.exit_signal:
           
            image = gs.rgb_frames.get()  
            image = cv2.resize(image, (640, 480))
            draw = image.copy()
            # Detect faces using Haar Cascade
            objects = self.cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
            detect_objects = []
            for (x, y, w, h) in objects:
                gray_face_roi = image[y:y+h, x:x+w]  # Extract grayscale face region
        
                # Preprocess the grayscale face image
                gray_face_resized = cv2.resize(gray_face_roi, (32, 32))
                gray_face_normalized = gray_face_resized.astype('float32') / 255.0
                gray_face_normalized = np.expand_dims(gray_face_normalized, axis=0)  # Add batch dimension

                # Perform prediction
                input_name = self.onnx_session.get_inputs()[0].name
                output_name = self.onnx_session.get_outputs()[0].name
                predictions = self.onnx_session.run([output_name], {input_name: gray_face_normalized})
                # print("pre: ", predictions)

                # Get predicted class and confidence
                predicted_class = np.argmax(predictions)
                # print("Predicted class:", predicted_class)
                confidence = predictions[0][0][predicted_class]
                # print("Confidence:", confidence)
                # print("Predicted class:", class_names[predicted_class])

                # Draw bounding box and label if confidence is more than 80%
                if self.classes[predicted_class] == "car" and confidence >= 0.93:
                    mid_x = x + w / 2
                    mid_y = y + h / 2
                    detect_objects.append((mid_x, mid_y))
                    
                    if draw is not None and gs.show_Object:
                        cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(draw, self.classes[predicted_class]+f' {str(round(confidence, 3))}', (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            # gs.objects = detect_objects
            
            if gs.show_Object:
                cv2.imshow('Detection object', draw)
                cv2.waitKey(1)
            
