import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import time
import threading

# Load the pre-trained model
model = load_model('/home/khanh/code/model.h5')

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier('/home/khanh/code/haarcascade_frontalface_default.xml')

# Preprocess the image before passing to the model
def preprocess_image(image, target_size=(48, 48)):
    image = cv2.resize(image, target_size)                
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)       
    image = cv2.equalizeHist(image)                       
    image = image / 255.0                                 
    image = np.expand_dims(image, axis=-1)                
    image = np.expand_dims(image, axis=0)                
    return image

# Function to predict age and gender from a preprocessed face image
def predict_age_gender(image):
    processed_image = preprocess_image(image)           
    age_pred, gender_pred = model.predict(processed_image) 
    age = age_pred[0][0]                                
    gender = np.argmax(gender_pred[0])                  
    return int(age), gender  

# Thread for capturing video
class VideoCaptureThread(threading.Thread):
    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, self.frame = self.cap.read()
            if not ret:
                print("Cannot read from webcam.")
                break

    def stop(self):
        self.running = False
        self.cap.release()

def detect_and_predict_from_webcam():
    video_thread = VideoCaptureThread()
    video_thread.start()

    # Variables to store previous predictions, and queues for averaging
    prev_age, prev_gender = None, None
    age_queue = deque(maxlen=5)  
    gender_queue = deque(maxlen=5)  
    frames_counter = 0
    update_interval = 3 

    # FPS calculation
    prev_time = time.time()

    while True:
        if video_thread.frame is not None:
            frame = video_thread.frame.copy()
            frame = cv2.flip(frame, 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Increment frame counter
            frames_counter += 1

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                if frames_counter % update_interval == 0 or prev_age is None:
                    age, gender = predict_age_gender(face)
                    age_queue.append(age)                    
                    gender_queue.append(gender)              
        
                    prev_age = int(np.mean(age_queue))      
                    prev_gender = int(np.mean(gender_queue)) 

                # Display the last averaged age and gender
                gender_text = 'Male' if prev_gender == 0 else 'Female'
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around face
                cv2.putText(frame, f'Age: {prev_age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(frame, f'Gender: {gender_text}', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Show FPS on the frame
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show the frame with predictions and FPS
            cv2.imshow('Face Age and Gender Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    video_thread.stop()
    cv2.destroyAllWindows()

detect_and_predict_from_webcam()
