import mediapipe as mp
import cv2
import numpy as np
import HandTrackingModule as htm


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

#--------------
detector=htm.handDetector(detectionCon=0)
tipIds=[4,8,12,16,20]  # thumb , index , middle , ring , pinky
totalFingers=0
flag=False

detect=[0,0,0,0,0,0,0,0]

#--------------------

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    ct=0
    while cap.isOpened():
        ret, frame = cap.read()

       
        sucess,img=cap.read()
        img=detector.findHands(img)
        lmList=detector.findPosition(img,draw =False)
        if len(lmList)!=0:
            fingers=[]

            #thumb
            if lmList[tipIds[0]][1]>lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            #4 fingers
            for id in range(1,5):
                if lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # print(fingers)
            totalFingers= fingers.count(1)

            # print(totalFingers)
            thumb_up=lmList[tipIds[0]][1]
            thumb_down=lmList[tipIds[0]-1][1]
            thumb_ans=thumb_up>thumb_down

            index_up=lmList[tipIds[1]][2]
            index_down=lmList[tipIds[1]-2][2]
            index_ans=index_up<index_down

            middle_up=lmList[tipIds[2]][2]
            middle_down=lmList[tipIds[2]-2][2]
            middle_ans=middle_up<middle_down

            ring_up=lmList[tipIds[3]][2]
            ring_down=lmList[tipIds[3]-2][2]
            ring_ans=ring_up<ring_down

            pinky_up=lmList[tipIds[4]][2]
            pinky_down=lmList[tipIds[4]-2][2]
            pinky_ans=pinky_up<pinky_down

            if fingers ==[1,0,0,0,1] and ct>10:
                print ("Emergency")
                ct=0
            ct+=1
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()