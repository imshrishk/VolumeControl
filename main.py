import cv2
import numpy as np
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui
import time
import threading

class GestureController:
    def __init__(self):
        self.wCam, self.hCam = 640, 480
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        
        self.pTime = 0
        self.cTime = 0
        
        # Audio setup
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.volRange = self.volume.GetVolumeRange()
        self.minVol, self.maxVol = self.volRange[0], self.volRange[1]
        
        # Gesture states
        self.volume_control_active = False
        self.brightness_control_active = False
        self.scroll_mode_active = False
        
        # Smoothing
        self.smoothing_factor = 0.5
        self.prev_vol = self.volume.GetMasterVolumeLevel()
        self.prev_brightness = 50  # Assuming initial brightness of 50%
        
        # Multi-threading for non-blocking operations
        self.thread_active = True
        self.thread = threading.Thread(target=self.background_tasks)
        self.thread.start()

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        return lm_list

    def calculate_hand_size(self, lm_list):
        """Calculate hand size to normalize distances based on wrist to middle finger base distance."""
        if len(lm_list) == 0:
            return None
        wrist = lm_list[0]  # Wrist
        middle_finger_base = lm_list[9]  # Base of middle finger
        hand_size = math.hypot(middle_finger_base[1] - wrist[1], middle_finger_base[2] - wrist[2])
        return hand_size

    def detect_gestures(self, lm_list):
        if len(lm_list) != 0:
            hand_size = self.calculate_hand_size(lm_list)
            if not hand_size:
                return self.img, None

            # Normalize the length with respect to hand size
            thumb_tip = lm_list[4]
            index_tip = lm_list[8]
            length = math.hypot(index_tip[1] - thumb_tip[1], index_tip[2] - thumb_tip[2]) / hand_size

            if 0.2 < length < 1.0:  # Normalized distances
                self.volume_control_active = True
                self.brightness_control_active = False
                return self.handle_volume_control(length)
            
            # Brightness control gesture (thumb and middle finger distance)
            middle_tip = lm_list[12]
            length_brightness = math.hypot(middle_tip[1] - thumb_tip[1], middle_tip[2] - thumb_tip[2]) / hand_size
            
            if 0.2 < length_brightness < 1.0:  # Normalized distances
                self.brightness_control_active = True
                self.volume_control_active = False
                return self.handle_brightness_control(length_brightness)
            
            # Scroll mode gesture (index and middle finger extended)
            if lm_list[8][2] < lm_list[6][2] and lm_list[12][2] < lm_list[10][2]:
                self.scroll_mode_active = True
                return self.handle_scroll_mode(lm_list[8][2])
            else:
                self.scroll_mode_active = False
            
            # Reset gesture (open palm)
            if all(lm_list[tip][2] < lm_list[tip - 2][2] for tip in [8, 12, 16, 20]):
                self.volume_control_active = False
                self.brightness_control_active = False
                self.scroll_mode_active = False
        
        return self.img, None

    def handle_volume_control(self, length):
        vol = np.interp(length, [0.2, 1.0], [self.minVol, self.maxVol])
        vol = self.smoothing_factor * vol + (1 - self.smoothing_factor) * self.prev_vol
        self.volume.SetMasterVolumeLevel(vol, None)
        self.prev_vol = vol
        
        volBar = np.interp(vol, [self.minVol, self.maxVol], [400, 150])
        volPer = np.interp(vol, [self.minVol, self.maxVol], [0, 100])
        
        cv2.rectangle(self.img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(self.img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(self.img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
        
        return self.img, "Volume"

    def handle_brightness_control(self, length):
        brightness = np.interp(length, [0.2, 1.0], [0, 100])
        brightness = self.smoothing_factor * brightness + (1 - self.smoothing_factor) * self.prev_brightness
        self.prev_brightness = brightness
        
        # Set system brightness (This is OS-dependent and may require additional libraries)
        # screen_brightness_control.set_brightness(int(brightness))
        
        cv2.rectangle(self.img, (50, 150), (85, 400), (255, 255, 0), 3)
        cv2.rectangle(self.img, (50, int(400 - brightness * 2.5)), (85, 400), (255, 255, 0), cv2.FILLED)
        cv2.putText(self.img, f'{int(brightness)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)
        
        return self.img, "Brightness"

    def handle_scroll_mode(self, y):
        scroll_speed = np.interp(y, [100, 300], [-20, 20])
        pyautogui.scroll(int(scroll_speed))
        cv2.putText(self.img, "Scroll Mode", (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        return self.img, "Scroll"

    def background_tasks(self):
        while self.thread_active:
            # Perform any background tasks here
            time.sleep(0.1)

    def run(self):
        while True:
            success, self.img = self.cap.read()
            self.img = cv2.flip(self.img, 1)
            
            self.img = self.find_hands(self.img)
            lm_list = self.find_position(self.img)
            
            self.img, gesture = self.detect_gestures(lm_list)

            # FPS calculation
            self.cTime = time.time()
            fps = 1 / (self.cTime - self.pTime)
            self.pTime = self.cTime
            
            cv2.putText(self.img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            cv2.imshow("Image", self.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.thread_active = False
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = GestureController()
    controller.run()
