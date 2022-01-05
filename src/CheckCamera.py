import cv2
from numpy import asarray
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

def camera():

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        _,image = cam.read()
        pixels = asarray(image)
    
        results = detector.detect_faces(pixels)

        for f in results:
            x, y, w, h =  f['box']
            face = image[y:y+h, x:x+w]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('CheckCamera', image)
        if (cv2.waitKey(1) == ord('q')):
            break
    
    # print("Attendance Successful")
    cam.release()
    cv2.destroyAllWindows()