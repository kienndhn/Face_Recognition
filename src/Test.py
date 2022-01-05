import datetime
import os
import time

import cv2
import pandas as pd

recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
recognizer.read("./TrainingImageLabel/Trainner.yml")

def main():
    while True:
        path=input("Input Image path: ")
        # path="./Test/1695842.jpg"
        recognize_attendence(path)
        # image = cv2.imread(".\\Training\\Adele\\394839ea00000578_0_image_m_86_1529881910097_donp.jpg")
        
#-------------------------
def recognize_attendence(path):
    
    harcascadePath = "./haarcascade/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize and start realtime video capture
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    df = pd.read_csv("StudentDetails"+os.sep+"StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5,flags = cv2.CASCADE_SCALE_IMAGE)
    for(x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (10, 159, 255), 2)
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            
        if conf < 100:
            aa = df.loc[df['Id'] == Id]['Name'].values
            confstr = "  {0}%".format(round(100 - conf))
            tt = str(Id)+"-"+aa

        else:
            Id = '  Unknown  '
            tt = str(Id)
            confstr = "  {0}%".format(round(100 - conf))

        if (100-conf) > 67:
            ts = time.time()
            # date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            # timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            aa = str(aa)[2:-2]
            # attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]

        tt = str(tt)[2:-2]
        if(100-conf) > 67:
            tt = tt + " [Pass]"
            cv2.putText(image, str(tt), (x+5,y-5), font, 1, (255, 255, 255), 2)
        else:
            cv2.putText(image, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

        if (100-conf) > 67:
            cv2.putText(image, str(confstr), (x + 5, y + h - 5), font,1, (0, 255, 0),1 )
        elif (100-conf) > 50:
            cv2.putText(image, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
        else:
            cv2.putText(image, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)



    cv2.imshow('Attendance', image)
    # ts = time.time()
    # date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    # timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    # Hour, Minute, Second = timeStamp.split(":")
    # fileName = "Attendance"+os.sep+"Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()