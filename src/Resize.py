import cv2
import os



harcascadePath = "./haarcascade/haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(harcascadePath)


paths = [os.path.join("Face", f) for f in os.listdir("Face")]

id=0
for path in paths:
    # id = id + 1
    sampleNum = 0
    directory = "Resize"+os.sep+os.path.split(path)[-1]+"."+str(id)
    if not os.path.exists(directory):
        os.makedirs(directory)

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # # print(imagePaths)
    for imgPath in imagePaths:
        img = cv2.imread(imgPath)
        resize = cv2.resize(img, (160, 160))
        sampleNum = sampleNum+1
        cv2.imwrite(directory+os.sep +str(sampleNum) + ".jpg", resize)
        
    #     if img is not None:
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    #         # print(type(faces))
    #         for(x, y, w, h) in faces:
    #             cv2.rectangle(img, (x, y), (x+w, y+h), (10, 159, 255), 2)
    #         # incrementing sample number
                
    #         # saving the captured face in the dataset folder TrainingImage
    #             if(w >= 60 and h >= 60): 
    #                 # print(w, h)
    #                 sampleNum = sampleNum+1
    #                 cv2.imwrite(directory+os.sep +str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
    #         # display the frame
    print(path)
            
    

