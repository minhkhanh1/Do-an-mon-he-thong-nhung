import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import collections
from collections import deque
from queue import Queue  # Cập nhật ở đây
import time
import threading

emotion_model = load_model('/home/khanh/code/modelv1.h5')
face_cascade = cv2.CascadeClassifier('/home/khanh/code/haarcascade_frontalface_default.xml')

emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Hàng đợi cho việc dự đoán cảm xúc
emotion_queue = Queue(maxsize=10)
predicted_emotion = "Uncertain"
emotion_lock = threading.Lock()

def preprocess_frame_for_emotion(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray, (48, 48))
    normalized_img = resized_img.astype('float32') / 255.0
    reshaped_img = np.expand_dims(normalized_img, axis=-1)
    reshaped_img = np.expand_dims(reshaped_img, axis=0)
    return reshaped_img

def smooth_emotion_predictions(predictions):
    emotion_predictions_queue.append(predictions)
    avg_predictions = np.mean(emotion_predictions_queue, axis=0)
    return avg_predictions

def predict_emotion(face):
    preprocessed_face_emotion = preprocess_frame_for_emotion(face)
    predictions_emotion = emotion_model.predict(preprocessed_face_emotion)
    return predictions_emotion

# Luồng xử lý cảm xúc
def emotion_thread():
    global predicted_emotion
    while True:
        face = emotion_queue.get()  # Lấy khuôn mặt từ hàng đợi
        if face is None:
            break  # Thoát khi nhận được tín hiệu dừng
        predictions_emotion = predict_emotion(face)
        max_prob = np.max(predictions_emotion)
        predicted_class = np.argmax(predictions_emotion)

        if max_prob > confidence_threshold:
            predicted_emotion_local = emotion_labels[predicted_class]
        else:
            predicted_emotion_local = "Uncertain"
        
        with emotion_lock:
            predicted_emotion = predicted_emotion_local
        emotion_queue.task_done()  # Đánh dấu công việc đã hoàn thành

# Bắt đầu luồng xử lý cảm xúc
emotion_processor_thread = threading.Thread(target=emotion_thread)
emotion_processor_thread.start()

# Bắt đầu video capture từ webcam
cap = cv2.VideoCapture(0)  # 0 là camera mặc định

# Thiết lập độ phân giải
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

confidence_threshold = 0.6
frames_counter = 0

# Biến để tính FPS
fps_start_time = time.time()
fps_display_interval = 1
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=7)
    frames_counter += 1
    frame_count += 1

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        if frames_counter % 5 == 0:
            if not emotion_queue.full():
                emotion_queue.put(face)  

        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        with emotion_lock:
            current_emotion = predicted_emotion

        cv2.putText(frame, f'Emotion: {current_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if (time.time() - fps_start_time) > fps_display_interval:
        fps = frame_count / (time.time() - fps_start_time)
        fps_start_time = time.time()
        frame_count = 0

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
  
    cv2.imshow('Emotion Recognition', frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

emotion_queue.put(None)  
emotion_processor_thread.join()  

cap.release()
cv2.destroyAllWindows()
