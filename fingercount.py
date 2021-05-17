import cv2
import handtrackingmodule as htm
import os


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fps=cap.get(cv2.CAP_PROP_FPS)

folder = "fingerimages"
mylist = os.listdir(folder)
print(mylist)

overlay = []
for impath in mylist:
    image = cv2.imread(f'{folder}/{impath}')
    # print(f'{folder}/{impath}')
    overlay.append(image)

detector=htm.handdetector(maxhands=1,detectionconfidence=0.75)
tipid=[4,8,12,16,20]
# thumb , index, middle , ring , pinky

while True:
    _, frame = cap.read()

    #for a right hand,whole program

    frame=detector.findhands(frame)
    lmlist=detector.findposition(frame,draw=False)
    #print(lmlist)
    if len(lmlist)!=0:
        #trying to get tip , based on that whether hand open or closed
        #POINTS NEEDED: 4,8,12,16,20 , so if point 8(top of index) is lower
        #in position than point 6(mid of index)=> that means index is down

        # for every element in lm list : lmlist[index][xposition][yposition]
        fingers = []

        # for thumb , x position of tip and mid is checked
        if lmlist[tipid[0]][1] > lmlist[tipid[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #for each finger , y position of tip and mid is checked
        for id in range(1,5):
            if lmlist[tipid[id]][2]<lmlist[tipid[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)


        total_fingers=fingers.count(1)#frequency of occurance of 1

        #print(total_fingers)
        #now change image according to number of fingers
        h, w, c = overlay[total_fingers-1].shape
        frame[0:h,0:w]=overlay[total_fingers-1]#for last 0-1 is -1 that is last elmt

        cv2.putText(frame, "Fingers detected => "+str(total_fingers) , (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    else:
        cv2.putText(frame, "Hand Not Detected", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.putText(frame,str(fps)+" fps",(800,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
    cv2.imshow('live', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
