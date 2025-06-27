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
    min_detection_confidence=0.8,  # Increased detection confidence
    min_tracking_confidence=0.8  # Increased tracking confidence
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
smoothening = 7  # Increased smoothing for cursor movement
cursor_buffer = deque(maxlen=smoothening)  # For cursor position smoothing

# Thresholds (tuned for better responsiveness)
CLICK_DISTANCE = 0.045  # Reduced click threshold for faster response
SCROLL_ACTIVATION_DISTANCE = 0.07  # Slightly reduced for easier activation
SCROLL_SENSITIVITY = 100  # Default scroll sensitivity
MIN_SCROLL_SENSITIVITY = 50
MAX_SCROLL_SENSITIVITY = 200
SCROLL_DEADZONE = 0.008  # Reduced deadzone for more responsive scrolling

# States
clicked = False
scrolling = False
scroll_reference_y = 0
scroll_buffer = deque(maxlen=5)  # For scroll smoothing

# Performance tracking (for adaptive thresholds)
frame_count = 0
last_click_time = 0
click_cooldown = 10  # frames

# Cursor acceleration parameters
ACCELERATION_THRESHOLD = 0.02  # Distance threshold for acceleration
ACCELERATION_FACTOR = 1.5  # How much to accelerate movement beyond threshold

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
            cv2.circle(frame, (int(index_tip.x * frame_w), int(index_tip.y * frame_h)), 10, (0, 255, 0),
                       -1)  # Green for index
            cv2.circle(frame, (int(thumb_tip.x * frame_w), int(thumb_tip.y * frame_h)), 10, (0, 0, 255),
                       -1)  # Red for thumb
            cv2.circle(frame, (int(middle_tip.x * frame_w), int(middle_tip.y * frame_h)), 10, (255, 255, 0),
                       -1)  # Yellow for middle

            # Calculate distances between fingers
            thumb_index_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
            middle_index_dist = math.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)

            # Visual feedback
            mode_text = "Normal"
            text_color = (0, 255, 0)  # Green

            # Check for scroll activation (middle finger close to index)
            if middle_index_dist < SCROLL_ACTIVATION_DISTANCE:
                if not scrolling:  # Just entered scroll mode
                    scrolling = True
                    scroll_reference_y = (middle_tip.y + index_tip.y) / 2  # Average of both fingers
                    scroll_buffer.clear()
                    # Lock the cursor position when entering scroll mode
                    scroll_cursor_lock_x, scroll_cursor_lock_y = pg.position()

                mode_text = "Scroll Mode"
                text_color = (255, 255, 0)  # Yellow

                # Calculate current average finger position
                current_y = (middle_tip.y + index_tip.y) / 2

                # Calculate scroll amount based on vertical movement from reference
                scroll_diff = current_y - scroll_reference_y

                # Add to scroll buffer for smoothing
                scroll_buffer.append(scroll_diff)

                # Get smoothed scroll difference
                smoothed_diff = np.mean(scroll_buffer) if scroll_buffer else 0

                # Calculate scroll amount with sensitivity
                scroll_amount = smoothed_diff * SCROLL_SENSITIVITY

                # Apply scroll if movement exceeds deadzone
                if abs(scroll_amount) > SCROLL_DEADZONE * SCROLL_SENSITIVITY:
                    pg.scroll(int(-scroll_amount))  # Negative because screen scrolls opposite to finger
                    # Update reference position to current position for relative scrolling
                    scroll_reference_y = current_y

                    # Visual feedback - arrow showing scroll direction
                    arrow_length = 50 * smoothed_diff * SCROLL_SENSITIVITY / 100
                    center_x = int((index_tip.x + middle_tip.x) / 2 * frame_w)
                    center_y = int((index_tip.y + middle_tip.y) / 2 * frame_h)
                    cv2.arrowedLine(frame,
                                    (center_x, center_y),
                                    (center_x, center_y - int(arrow_length)),
                                    (255, 255, 0), 3, tipLength=0.3)

                # Keep cursor locked in scroll mode
                pg.moveTo(scroll_cursor_lock_x, scroll_cursor_lock_y)
            else:
                scrolling = False

                # Normal cursor movement when not scrolling
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
                        cv2.circle(frame,
                                   (int(index_tip.x * frame_w), int(index_tip.y * frame_h)),
                                   20, (0, 0, 255), 2)
                else:
                    clicked = False

            # Display mode and distances with better formatting
            cv2.putText(frame, f"Mode: {mode_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame, f"Thumb-Index: {thumb_index_dist:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 0) if thumb_index_dist > CLICK_DISTANCE else (0, 0, 255), 2)
            cv2.putText(frame, f"Middle-Index: {middle_index_dist:.3f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 0) if middle_index_dist > SCROLL_ACTIVATION_DISTANCE else (255, 255, 0), 2)
            cv2.putText(frame, f"Scroll Sens: {SCROLL_SENSITIVITY} (1/0 to adjust)", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Tip: Move fast for accelerated cursor", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Display the frame
    cv2.putText(frame, "Developed by Abuzar", (frame_w - 200, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Enhanced Hand Control (Index-Move, Thumb-Click, Pinch-Scroll)", frame)

    # Adjust scroll sensitivity with keyboard
    key = cv2.waitKey(1)
    if key == ord("s"):
        break
    elif key == ord("1") and SCROLL_SENSITIVITY < MAX_SCROLL_SENSITIVITY:
        SCROLL_SENSITIVITY += 10
    elif key == ord("0") and SCROLL_SENSITIVITY > MIN_SCROLL_SENSITIVITY:
        SCROLL_SENSITIVITY -= 10
    elif key == ord("r"):  # Reset sensitivity
        SCROLL_SENSITIVITY = 100

# Release resources
camera.release()
cv2.destroyAllWindows()