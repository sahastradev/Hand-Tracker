from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
import cv2
import time
import numpy as np
import HandModule as hm
import math

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
prevTime = 0
vol = 0
volBar = 400
volPerc = 0

detector = hm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

while True:
    success, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) != 0:
        # Assigning the landmark's position to the variables
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Drawubg circles on the two finger points
        cv2.circle(frame, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        vol = np.interp(length, [50, 190], [minVol, maxVol])
        volBar = np.interp(length, [50, 190], [400, 150])
        volPerc = np.interp(length, [50, 190], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)

        if length < 30:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(frame, (50, int(volBar)), (85, 400),
                  (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, f'{int(volPerc)} %', (40, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # put image on the screen and press 'q' key to quit
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
