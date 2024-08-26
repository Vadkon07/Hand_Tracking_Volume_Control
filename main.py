import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.75, min_tracking_confidence=0.75, max_num_hands=4)

# Get default audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

cap = cv2.VideoCapture(0)

# That's ROI coordinates, which you use to crop your frame distance
x, y, w, h = 100, 100, 1000, 200  # Adjust these values as needed, depends from your distance

# Volume control parameters
volume_increment = 0.10  # Adjust as needed
current_volume = volume.GetMasterVolumeLevel()

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1) #Comment it if your left hand was detected as a right hand, or left hand as a right hand. I commented it because my laptop decided to mirror his camera

    roi = img[y:y+h, x:x+w]

    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get the label of the hand (Left or Right)
            label = MessageToDict(handedness)['classification'][0]['label']

            if label == 'Left':
                # Decrease volume
                current_volume = max(-65.0, current_volume - volume_increment)  # -65 dB is minimum volume, you can change it
                volume.SetMasterVolumeLevel(current_volume, None)
                cv2.putText(roi,f'Decreased volume: {current_volume:.2f} dB', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif label == 'Right':
                # Increase volume
                current_volume = min(0.0, current_volume + volume_increment)  # 0 dB is maximum volume, you also can set a maximum level of volume here
                volume.SetMasterVolumeLevel(current_volume, None)
                cv2.putText(roi,f'Increased volume: {current_volume:.2f} dB', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("Both hands detected") #For now it doesn't work, solve it please

            # Display the label on the screen

            cv2.putText(roi, label + ' Hand', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Cropped Frame', roi) #Comment if you don't want to see a window of cv2
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()