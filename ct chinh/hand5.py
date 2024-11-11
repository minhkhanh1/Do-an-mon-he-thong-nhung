import cv2
import mediapipe as mp
import time
import math
import threading
import board
import neopixel

# NeoPixel setup
PIXEL_PIN = board.D18  # GPIO pin connected to the data input of the NeoPixel strip
NUM_PIXELS = 8  # Number of pixels in the strip
pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, auto_write=False)

# Function to set LED brightness
def set_led_brightness(level):
    if level == 'Low':
        brightness = 0.3
    elif level == 'Medium':
        brightness = 0.6
    elif level == 'High':
        brightness = 1.0
    else:
        brightness = 0.0

    color = (int(255 * brightness), int(255 * brightness), int(255 * brightness))  # RGB color
    for i in range(NUM_PIXELS):
        pixels[i] = color
    pixels.show()

class VideoStream:
    def __init__(self, src=0, width=320, height=240):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Function to send data periodically
def send_data_periodically():
    global last_output_time, current_bright_level, bright_mode, last_bright_level, led_state
    last_bright_level = None  # Store previous brightness level

    while True:
        time.sleep(0.5)  # Perform every 1 second
        current_time = time.time()

        if bright_mode:
            if current_bright_level != last_bright_level:
                print(f'Sending bright level: {current_bright_level}')
                set_led_brightness(current_bright_level)
                last_bright_level = current_bright_level  # Update sent bright level
        else:
            if last_bright_level != 'None':
                print('Sending bright level: None')
                last_bright_level = 'None'  

        last_output_time = current_time

# Mediapipe hand setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils  

# Video stream setup
video_stream = VideoStream().start()

# Start the data sending thread
current_bright_level = 'None'
bright_mode = False
led_state = False  # Initial LED state
last_output_time = time.time()  # Track the last time the data was output
data_thread = threading.Thread(target=send_data_periodically)
data_thread.daemon = True  # Ensures the thread will exit when the main program exits
data_thread.start()

prev_frame_time = 0

# Additional variables for delay
last_bright_mode_change_time = time.time()  # Track the last time bright mode was changed
last_led_toggle_time = time.time()  # Track the last time the LED was toggled

while True:
    ret, frame = video_stream.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip image vertically
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            distance_thumb_middle = calculate_distance(thumb_tip, middle_tip)
            distance_thumb_index = calculate_distance(thumb_tip, index_tip)

            # Toggle bright mode if thumb touches middle finger
            if distance_thumb_middle < 0.1:  # Adjusted threshold
                if current_time - last_bright_mode_change_time >= 1:  # 1 second delay
                    if not bright_mode:
                        bright_mode = True
                        display_text = 'Bright mode activated'
                        # Set LED to last known brightness level
                        set_led_brightness(current_bright_level)
                    else:
                        bright_mode = False
                        display_text = 'Bright mode deactivated'
                        # Maintain the current LED brightness level
                        set_led_brightness(current_bright_level)  # Keep the LED at its last brightness level

                    last_output_time = current_time  # Reset output timer
                    last_bright_mode_change_time = current_time  # Update the last bright mode change time
                    print(display_text)

            # Toggle LED if thumb touches index finger, but only when bright mode is OFF
            if not bright_mode and distance_thumb_index < 0.1:  # Adjusted threshold
                if current_time - last_led_toggle_time >= 1:  # 1 second delay for LED toggle
                    led_state = not led_state
                    if current_bright_level == 'None':
                        set_led_brightness('Low' if led_state else 'None')  # Set LED based on state
                    else:
                        set_led_brightness(current_bright_level if led_state else 'None')
                    last_output_time = current_time  # Reset output timer
                    last_led_toggle_time = current_time  # Update the last LED toggle time
                    print(f'LED toggled to: {"ON" if led_state else "OFF"}')

            # In bright mode, determine brightness level based on thumb-index distance
            if bright_mode:
                if distance_thumb_index < 0.15:
                    current_bright_level = 'Low'
                elif distance_thumb_index < 0.25:
                    current_bright_level = 'Medium'
                else:
                    current_bright_level = 'High'

   
    bright_text = f'Brightness Level: {current_bright_level} (off)' if not bright_mode else f'Brightness Level: {current_bright_level}'


    text_x = 10
    text_y = 20  

    cv2.putText(frame, bright_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.stop()
cv2.destroyAllWindows()
