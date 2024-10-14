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
import pygame
import speech_recognition as sr
from PIL import ImageGrab
import keyboard
import screen_brightness_control as sbc

class AdvancedGestureController:
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
        self.zoom_mode_active = False
        #self.drawing_mode_active = False
        
        # Smoothing
        self.smoothing_factor = 0.5
        self.prev_vol = self.volume.GetMasterVolumeLevel()
        self.prev_brightness = sbc.get_brightness()[0]
        
        # Drawing setup
        self.drawing_canvas = np.zeros((self.hCam, self.wCam, 3), np.uint8)
        self.drawing_color = (0, 255, 0)  # Green
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Music player setup
        pygame.mixer.init()
        self.current_track = 0
        self.tracks = ['path/to/song1.mp3', 'path/to/song2.mp3', 'path/to/song3.mp3']
        
        # Multi-threading for non-blocking operations
        self.thread_active = True
        self.thread = threading.Thread(target=self.background_tasks)
        self.thread.start()
        
        # Calibration
        self.calibrated = False
        self.min_hand_size = float('inf')
        self.max_hand_size = 0

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
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list

    def calculate_hand_size(self, lm_list):
        if len(lm_list) == 0:
            return None
        wrist = lm_list[0]
        middle_finger_tip = lm_list[12]
        hand_size = math.hypot(middle_finger_tip[1] - wrist[1], middle_finger_tip[2] - wrist[2])
        
        if not self.calibrated:
            self.min_hand_size = min(self.min_hand_size, hand_size)
            self.max_hand_size = max(self.max_hand_size, hand_size)
            if self.max_hand_size - self.min_hand_size > 100:  # Arbitrary threshold
                self.calibrated = True
        
        return hand_size

    def detect_gestures(self, lm_list):
        if len(lm_list) != 0:
            hand_size = self.calculate_hand_size(lm_list)
            if not hand_size:
                return self.img, None

            # Normalize distances
            thumb_tip = lm_list[4]
            index_tip = lm_list[8]
            middle_tip = lm_list[12]
            ring_tip = lm_list[16]
            pinky_tip = lm_list[20]

            # Volume control (thumb and index)
            volume_length = math.hypot(index_tip[1] - thumb_tip[1], index_tip[2] - thumb_tip[2]) / hand_size
            if 0.1 < volume_length < 0.5:
                self.volume_control_active = True
                return self.handle_volume_control(volume_length)
            
            # Brightness control (thumb and middle)
            brightness_length = math.hypot(middle_tip[1] - thumb_tip[1], middle_tip[2] - thumb_tip[2]) / hand_size
            if 0.1 < brightness_length < 0.5:
                self.brightness_control_active = True
                return self.handle_brightness_control(brightness_length)
            
            # Scroll mode (index and middle extended)
            if lm_list[8][2] < lm_list[6][2] and lm_list[12][2] < lm_list[10][2]:
                self.scroll_mode_active = True
                return self.handle_scroll_mode(lm_list[8][2])
            
            # Zoom mode (thumb, index, and middle extended)
            if all(lm_list[tip][2] < lm_list[tip - 2][2] for tip in [4, 8, 12]):
                self.zoom_mode_active = True
                return self.handle_zoom_mode(lm_list)
            
            # Drawing mode (index extended)
            #if lm_list[8][2] < lm_list[6][2]:
            #    self.drawing_mode_active = True
            #    return self.handle_drawing_mode(lm_list[8][1:])
            
            # Music control (thumb, index, middle, and ring extended)
            if all(lm_list[tip][2] < lm_list[tip - 2][2] for tip in [4, 8, 12, 16]):
                return self.handle_music_control()
            
            # Voice command mode (all fingers extended)
            if all(lm_list[tip][2] < lm_list[tip - 2][2] for tip in [8, 12, 16, 20]):
                return self.handle_voice_command()
            
            # Reset all modes
            self.volume_control_active = False
            self.brightness_control_active = False
            self.scroll_mode_active = False
            self.zoom_mode_active = False
            #self.drawing_mode_active = False
        
        return self.img, None

    def handle_volume_control(self, length):
        vol = np.interp(length, [0.1, 0.5], [self.minVol, self.maxVol])
        vol = self.smoothing_factor * vol + (1 - self.smoothing_factor) * self.prev_vol
        self.volume.SetMasterVolumeLevel(vol, None)
        self.prev_vol = vol
        
        volBar = np.interp(vol, [self.minVol, self.maxVol], [400, 150])
        volPer = np.interp(vol, [self.minVol, self.maxVol], [0, 100])
        
        cv2.rectangle(self.img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(self.img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(self.img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
        
        # Display current volume level
        current_vol = self.volume.GetMasterVolumeLevelScalar()
        cv2.putText(self.img, f'Current Volume: {int(current_vol * 100)}%', (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return self.img, "Volume"

    def handle_brightness_control(self, length):
        brightness = np.interp(length, [0.1, 0.5], [0, 100])
        brightness = self.smoothing_factor * brightness + (1 - self.smoothing_factor) * self.prev_brightness
        self.prev_brightness = brightness
        
        sbc.set_brightness(int(brightness))
        
        cv2.rectangle(self.img, (50, 150), (85, 400), (255, 255, 0), 3)
        cv2.rectangle(self.img, (50, int(400 - brightness * 2.5)), (85, 400), (255, 255, 0), cv2.FILLED)
        cv2.putText(self.img, f'{int(brightness)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)
        
        return self.img, "Brightness"

    def handle_scroll_mode(self, y):
        scroll_speed = np.interp(y, [100, 300], [-20, 20])
        pyautogui.scroll(int(scroll_speed))
        cv2.putText(self.img, "Scroll Mode", (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        return self.img, "Scroll"

    def handle_zoom_mode(self, lm_list):
        length = math.hypot(lm_list[4][1] - lm_list[8][1], lm_list[4][2] - lm_list[8][2])
        zoom_level = np.interp(length, [50, 200], [0, 20])
        keyboard.press_and_release('ctrl+' + str(int(zoom_level)))
        cv2.putText(self.img, f"Zoom: {int(zoom_level)}", (400, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
        return self.img, "Zoom"

    #def handle_drawing_mode(self, finger_tip):
    #    cv2.circle(self.drawing_canvas, finger_tip, 15, self.drawing_color, cv2.FILLED)
    #    self.img = cv2.addWeighted(self.img, 0.8, self.drawing_canvas, 0.2, 0)
    #    cv2.putText(self.img, "Drawing Mode", (400, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    #    return self.img, "Drawing"

    def handle_music_control(self):
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(self.tracks[self.current_track])
            pygame.mixer.music.play()
        else:
            pygame.mixer.music.stop()
            self.current_track = (self.current_track + 1) % len(self.tracks)
            pygame.mixer.music.load(self.tracks[self.current_track])
            pygame.mixer.music.play()
        
        cv2.putText(self.img, f"Playing Track {self.current_track + 1}", (300, 170), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
        return self.img, "Music"

    def handle_voice_command(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
        
        try:
            command = self.recognizer.recognize_google(audio).lower()
            if "screenshot" in command:
                self.take_screenshot()
            elif "clear drawing" in command:
                self.drawing_canvas = np.zeros((self.hCam, self.wCam, 3), np.uint8)
            # Add more voice commands here
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        
        cv2.putText(self.img, "Voice Command Mode", (300, 210), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)
        return self.img, "Voice"

    def take_screenshot(self):
        screenshot = ImageGrab.grab()
        screenshot.save("screenshot.png")
        cv2.putText(self.img, "Screenshot Taken!", (300, 250), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

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

            # Display calibration status
            if not self.calibrated:
                cv2.putText(self.img, "Calibrating... Move hand closer and farther", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(self.img, "Calibrated", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display active mode
            if gesture:
                cv2.putText(self.img, f"Active Mode: {gesture}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Gesture Control", self.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.thread_active = False
        self.cap.release()
        cv2.destroyAllWindows()

    def change_drawing_color(self):
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        self.drawing_color = colors[(colors.index(self.drawing_color) + 1) % len(colors)]

    def clear_drawing(self):
        self.drawing_canvas = np.zeros((self.hCam, self.wCam, 3), np.uint8)

    def save_drawing(self):
        cv2.imwrite("drawing.png", self.drawing_canvas)
        cv2.putText(self.img, "Drawing Saved!", (300, 290), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    def handle_pinch_gesture(self, lm_list):
        pinch_threshold = 30
        if math.hypot(lm_list[8][1] - lm_list[4][1], lm_list[8][2] - lm_list[4][2]) < pinch_threshold:
            x, y = (lm_list[8][1] + lm_list[4][1]) // 2, (lm_list[8][2] + lm_list[4][2]) // 2
            pyautogui.moveTo(x * 2, y * 2)  # Adjust for screen size difference
            return True
        return False

    def handle_gesture_shortcuts(self, lm_list):
        # Implement various gestures for different actions
        # Example: Thumb up for volume up, thumb down for volume down
        thumb_tip = lm_list[4][2]
        thumb_ip = lm_list[3][2]
        
        if thumb_tip < thumb_ip - 20:  # Thumb up
            pyautogui.press('volumeup')
            cv2.putText(self.img, "Volume Up", (300, 330), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
        elif thumb_tip > thumb_ip + 20:  # Thumb down
            pyautogui.press('volumedown')
            cv2.putText(self.img, "Volume Down", (300, 330), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    def text_to_speech(self, text):
        # Implement text-to-speech functionality
        # You can use libraries like pyttsx3 or gTTS for this feature
        pass

    def gesture_typing(self, lm_list):
        # Implement a gesture-based typing system
        # Map hand positions to letters or use a predictive text system
        pass

    def handle_face_detection(self, img):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img

    def handle_emotion_detection(self, img):
        # Implement emotion detection using a pre-trained model
        # You can use libraries like fer (Facial Expression Recognition)
        pass

    def handle_augmented_reality(self, img, lm_list):
        # Implement simple AR features
        # Example: Draw a virtual object on the palm
        if len(lm_list) > 0:
            palm_center = ((lm_list[0][1] + lm_list[9][1]) // 2, (lm_list[0][2] + lm_list[9][2]) // 2)
            cv2.circle(img, palm_center, 30, (0, 255, 255), -1)
        return img

    def handle_gesture_game(self, lm_list):
        # Implement a simple gesture-based game
        # Example: Rock-Paper-Scissors against the computer
        pass

if __name__ == "__main__":
    controller = AdvancedGestureController()
    controller.run()
