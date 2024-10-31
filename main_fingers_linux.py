import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.90, min_tracking_confidence=0.9, max_num_hands=2)

cap = cv2.VideoCapture(0)

# Define ROI if needed (currently commented out)
# x, y, w, h = 100, 100, 1000, 200

volume_increment = 0.05  # Adjust increment as needed

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    roi = img  # Use the whole image or uncomment the next line to use the ROI
    # roi = img[y:y+h, x:x+w]
    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            points = []
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * roi.shape[1]), int(lm.y * roi.shape[0])
                points.append((cx, cy))

            if len(points) > 1:
                points = np.array(points, np.int32)
                points = points.reshape((-1, 1, 2))

            label = "Hand"

            cv2.putText(roi, label + ' Hand', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.polylines(roi, [points], isClosed=False, color=(255, 255, 255), thickness=2)

            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * roi.shape[1]), int(lm.y * roi.shape[0])
                cv2.circle(roi, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                cv2.putText(roi, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            thumb_tip = hand_landmarks.landmark[4].y
            index_tip = hand_landmarks.landmark[8].y
            middle_tip = hand_landmarks.landmark[12].y
            ring_tip = hand_landmarks.landmark[16].y
            pinky_tip = hand_landmarks.landmark[20].y

            if index_tip < middle_tip and index_tip < thumb_tip and index_tip < ring_tip and index_tip < pinky_tip:
                cv2.putText(roi, 'Increased finger', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                os.system('amixer -D pulse sset Master 1%+')

            if index_tip > thumb_tip and index_tip > middle_tip and index_tip > ring_tip and index_tip > pinky_tip:
                cv2.putText(roi, 'Decreased finger', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                os.system('amixer -D pulse sset Master 1%-')

    cv2.imshow('Frame', roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

