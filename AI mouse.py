import cv2
import mediapipe as mp
import numpy as np
import time
import autopy 
import pyautogui  
import handtrackingmodule as htm
from pynput.mouse import Button, Controller

mouse = Controller()
cap = cv2.VideoCapture(0)

# Set appropriate webcam resolution
wcam = 640
hcam = 480
cap.set(3, wcam)
cap.set(4, hcam)

# Model setup
mphand = mp.solutions.hands
hands = mphand.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpdraw = mp.solutions.drawing_utils

ptime = 0

def find_fingertip(results):
    if results.multi_hand_landmarks:
        handlandmarks = results.multi_hand_landmarks[0]
        return handlandmarks.landmark[mphand.HandLandmark.INDEX_FINGER_TIP]
    return None

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * autopy.screen.size()[0])
        y = int(index_finger_tip.y * autopy.screen.size()[1])
        pyautogui.moveTo(x, y)

def left_click(landmarklist, thumb_index_distance):
    return (htm.calculate_angle(landmarklist[5], landmarklist[6], landmarklist[8]) < 50 and 
            htm.calculate_angle(landmarklist[9], landmarklist[10], landmarklist[12]) > 90 and 
            thumb_index_distance > 50)

def right_click(landmarklist, thumb_index_distance):
    return (htm.calculate_angle(landmarklist[9], landmarklist[10], landmarklist[12]) < 50 and 
            htm.calculate_angle(landmarklist[5], landmarklist[6], landmarklist[8]) > 90 and 
            thumb_index_distance > 50)

def double_click(landmarklist, thumb_index_distance):
    return (htm.calculate_angle(landmarklist[5], landmarklist[6], landmarklist[8]) < 50 and 
            htm.calculate_angle(landmarklist[9], landmarklist[10], landmarklist[12]) < 50 and 
            thumb_index_distance > 50)

def screenshot(landmarklist, thumb_index_distance):
    return (htm.calculate_angle(landmarklist[5], landmarklist[6], landmarklist[8]) < 50 and 
            htm.calculate_angle(landmarklist[9], landmarklist[10], landmarklist[12]) < 50 and 
            thumb_index_distance < 50)

def detect_gestures(img, landmarklist, results):
    if len(landmarklist) >= 21:
        index_finger_tip = find_fingertip(results)
        thumb_index_distance = htm.getdistance([landmarklist[4], landmarklist[5]])

        if thumb_index_distance < 50 and htm.calculate_angle(landmarklist[5], landmarklist[6], landmarklist[8]) > 90:
            move_mouse(index_finger_tip)

        elif left_click(landmarklist, thumb_index_distance):
            mouse.press(Button.left)
            mouse.release(Button.left)
        
        elif right_click(landmarklist, thumb_index_distance):
            mouse.press(Button.right)
            mouse.release(Button.right)

        elif double_click(landmarklist, thumb_index_distance):
            pyautogui.doubleClick()

        elif screenshot(landmarklist, thumb_index_distance):
            pyautogui.screenshot("gesture_screenshot.png")

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    landmarklist = []
    if results.multi_hand_landmarks:
        handlandmarks = results.multi_hand_landmarks[0]
        mpdraw.draw_landmarks(img, handlandmarks, mphand.HAND_CONNECTIONS)
        
        for lm in handlandmarks.landmark:
            landmarklist.append(lm)
      
    detect_gestures(img, landmarklist, results)

    ctime = time.time()
    fps = 1 / (ctime - ptime) if ctime - ptime != 0 else 0
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)   
    cv2.imshow("Image", img)
    cv2.waitKey(1)
