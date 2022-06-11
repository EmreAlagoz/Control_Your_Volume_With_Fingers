import cv2 as cv
import numpy as np
import time
import math
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Defining camera size
wCam , hCam = 640,480

# Defining camera
vid = cv.VideoCapture(0)
vid.set(3,wCam)
vid.set(4,hCam)

# Hand detection
detector = htm.handDetector(min_detection_confidence=0.8)

# FPS
currentTime = 0
previousTime = 0

# Audio management
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volumeRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
minVol = volumeRange[0]
maxVol = volumeRange[1]

while True:
    success, img = vid.read()

    # Find hands
    img = detector.findHands(img)

    # Finding landmarks
    lmList = detector.findPos(img,isDraw=False)
    if len(lmList)!=0:
        # Head finger
        x1,y1 = lmList[4][1],lmList[4][2]
        # Pointer finger
        x2,y2 = lmList[8][1],lmList[8][2]
        # Center of line
        centerX,centerY = (x1+x2)//2,(y1+y2)//2

        cv.circle(img,(x1,y1),7,(255,0,0),cv.FILLED)
        cv.circle(img,(x2,y2),7,(255,0,0),cv.FILLED)
        cv.circle(img, (centerX, centerY), 7, (255, 0, 0), cv.FILLED)
        cv.line(img,(x1,y1),(x2,y2),(255, 0, 0),5)

        # Distance between points
        dist = math.hypot(math.fabs(x1-x2),math.fabs(y1-y2))
        # Distance between point are min 20 and max 300.
        # Volume range is -64 to 0
        vol = np.interp(dist,[20,275],[minVol,maxVol])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)



    # FPS calculating
    currentTime = time.time()
    fps = (1/(currentTime-previousTime))
    fps = int(fps)
    previousTime = currentTime
    cv.putText(img, str(fps), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    cv.imshow('',img)