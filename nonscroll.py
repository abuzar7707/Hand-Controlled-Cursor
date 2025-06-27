import cv2
import mediapipe as mp
import pyautogui as pg
import math
import numpy as np
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8, #80&
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# Initialize camera with higher resolution if available
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Failed to connect to camera!")
    exit()

# Try to set higher resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get screen size
screen_w, screen_h = pg.size()

# Smoothing parameters
smoothening = 7
cursor_buffer = deque(maxlen=smoothening)

# Click threshold
CLICK_DISTANCE = 0.035

# States
clicked = False

# Performance tracking
frame_count = 0
last_click_time = 0
click_cooldown = 10  # frames

# Cursor acceleration parameters
ACCELERATION_THRESHOLD = 0.02
ACCELERATION_FACTOR = 1.5

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture frame!")
        break

    frame_count += 1

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks with different colors for key points
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Highlight important landmarks
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Draw colored circles on key points
            cv2.circle(frame, (int(index_tip.x * frame_w), int(index_tip.y * frame_h)), 10, (0, 255, 0), -1)  # Green for index
            cv2.circle(frame, (int(thumb_tip.x * frame_w), int(thumb_tip.y * frame_h)), 10, (0, 0, 255), -1)  # Red for thumb
            cv2.circle(frame, (int(middle_tip.x * frame_w), int(middle_tip.y * frame_h)), 10, (255, 255, 0), -1)  # Yellow for middle

            # Calculate distance between thumb and index
            thumb_index_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

            # Visual feedback
            mode_text = "Normal"
            text_color = (0, 255, 0)  # Green

            # Normal cursor movement
            screen_x = int(index_tip.x * screen_w)
            screen_y = int(index_tip.y * screen_h)

            # Calculate distance from previous position for acceleration
            if cursor_buffer:
                last_x, last_y = cursor_buffer[-1]
                dist = math.hypot(screen_x - last_x, screen_y - last_y) / screen_w

                # Apply acceleration if movement is large
                if dist > ACCELERATION_THRESHOLD:
                    screen_x = last_x + (screen_x - last_x) * ACCELERATION_FACTOR
                    screen_y = last_y + (screen_y - last_y) * ACCELERATION_FACTOR

            # Add to cursor buffer for smoothing
            cursor_buffer.append((screen_x, screen_y))

            # Get smoothed cursor position
            if cursor_buffer:
                smooth_x = int(np.mean([x for x, y in cursor_buffer]))
                smooth_y = int(np.mean([y for x, y in cursor_buffer]))

                # Move the cursor
                pg.moveTo(smooth_x, smooth_y)

            # Check for click (thumb close to index)
            if thumb_index_dist < CLICK_DISTANCE:
                if not clicked and (frame_count - last_click_time) > click_cooldown:
                    pg.click()
                    clicked = True
                    last_click_time = frame_count
                    mode_text = "Click"
                    text_color = (0, 0, 255)  # Red
                    # Visual feedback
                    cv2.circle(frame, (int(index_tip.x * frame_w), int(index_tip.y * frame_h)), 20, (0, 0, 255), 2)
            else:
                clicked = False

            # Display mode and distances
            cv2.putText(frame, f"Mode: {mode_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame, f"Thumb-Index: {thumb_index_dist:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 0) if thumb_index_dist > CLICK_DISTANCE else (0, 0, 255), 2)
            cv2.putText(frame, "Tip: Move fast for accelerated cursor", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Display the frame
    cv2.putText(frame, "Developed by Abuzar", (frame_w - 200, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Hand Controlled Cursor (Index-Move, Thumb-Click)", frame)

    # Exit on 's' key press
    if cv2.waitKey(1) == ord("s"):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()