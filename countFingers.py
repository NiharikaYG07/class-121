import cv2
import mediapipe as mp

cam=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
mpDrawing=mp.solutions.drawing_utils

hands=mpHands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5)
tipIds=[4,8,12,16,20]

def drawLandmarks(frame,hl):
    if hl:
        for i in hl:
            mpDrawing.draw_landmarks(frame,i,mpHands.HAND_CONNECTIONS)

def countFingers(frame,hl,handNo=0):
    if hl:
        landmarks=hl[0].landmark
        fingers=[]
        for i in tipIds:
            fingerTipY=landmarks[i].y
            fingerBottomY=landmarks[i-2].y
            thumbTipX=landmarks[i].x
            thumbBottomX=landmarks[i-2].x
            thumbTipY=landmarks[i].y
            thumbBottomY=landmarks[i-2].y

            if i!=4:
                if fingerTipY<fingerBottomY:
                    fingers.append(1)
                if fingerTipY>fingerBottomY:
                    fingers.append(0)
            else:
                if thumbTipX>thumbBottomX:
                    fingers.append(1)
                if thumbTipX<thumbBottomX:
                    fingers.append(0)
                if thumbTipY>thumbBottomY:
                    cv2.putText(frame,"disliked",(50,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
                if thumbTipY<thumbBottomY:
                    cv2.putText(frame,"liked",(50,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0,),2)
        
        totalFingers=fingers.count(1)
        msg=f"finger:{totalFingers}"
        cv2.putText(frame,msg,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

while True:
    ret,frame=cam.read()
    frame=cv2.flip(frame,1)
    results=hands.process(frame)
    handLandmarks=results.multi_hand_landmarks
    drawLandmarks(frame,handLandmarks)
    countFingers(frame,handLandmarks)

    cv2.imshow("camera",frame)

    if cv2.waitKey(2)==32:
        break

cv2.destroyAllWindows()