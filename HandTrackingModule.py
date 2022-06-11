import cv2 as cv
import numpy as np
import mediapipe as mp
import time


class handDetector():
    def __init__(self,
               static_image_mode=False,
               max_num_hands=2,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode,self.max_num_hands,self.min_detection_confidence,self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

        self.idNums = [8,12,16,20]


    def findHands(self,img,draw=True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks , self.mpHands.HAND_CONNECTIONS)
        return img


    def findPos(self,img, handNo = 0, isDraw=True):
        landmarksList = []
        if self.results.multi_hand_landmarks:
            selectedHand = self.results.multi_hand_landmarks[handNo]

            for id, landmark in enumerate(selectedHand.landmark):
                w, h, c = img.shape
                centerX, centerY = int(landmark.x * h), int(landmark.y * w)
                # print(id,centerX,centerY)
                landmarksList.append([id,centerX,centerY])
                if isDraw:
                    cv.circle(img,(centerX,centerY),15,(0,255,0),cv.FILLED)
        return landmarksList

    def fingersUp(self,img,isFlipped=False):
        fingers = []
        lmList = self.findPos(img,isDraw=False)
        if len(lmList) != 0:
            # Thumb
            if isFlipped:
                if lmList[4][1] < lmList[4 - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if lmList[4][1] > lmList[4 - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in self.idNums:
                if lmList[id][2] < lmList[id - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

def main():
    vid = cv.VideoCapture(0)
    previousTime = 0
    currentTime = 0

    detector = handDetector()

    while True:
        success, img = vid.read()
        img = detector.findHands(img)

        landmarkList = detector.findPos(img)
        if len(landmarkList)!=0:
            print(landmarkList[4])

        currentTime = time.time()
        fps = (1 / (currentTime - previousTime))
        fps = int(fps)
        previousTime = currentTime
        cv.putText(img, str(fps), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        cv.imshow('img', img)

if __name__ == '__main__':
    main()