import cv2    # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import pickle
from keras.models import load_model
from numpy import asarray
from numpy import expand_dims
from numpy import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from mtcnn.mtcnn import MTCNN

SVMModel = pickle.load(open("../SVMModel.sav", 'rb'))

FacenetModel = load_model('../facenet_keras.h5')

in_encoder = Normalizer(norm='l2')

data = load('../faceEmbedding-10samplesq.npz')
X, y= data['arr_0'],data['arr_1']

# X = in_encoder.transform(X)
out_encoder = LabelEncoder()
out_encoder.fit(y)
y = out_encoder.transform(y)
detector = MTCNN()
font = cv2.FONT_HERSHEY_SIMPLEX

def get_embedding(face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = FacenetModel.predict(samples)
	return yhat[0]


def face_predict(face_embedded):
    sample = expand_dims(face_embedded, axis=0)
    sample = in_encoder.transform(sample)
    # print(sample.shape)
    yhat_class = SVMModel.predict(sample)
    yhat_prob = SVMModel.predict_proba(sample)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_name = out_encoder.inverse_transform(yhat_class)
    return predict_name[0], class_probability

#-------------------------
def liveRecognition():

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        _,image = cam.read()
        pixels = asarray(image)
    
        results = detector.detect_faces(pixels)

        for f in results:
            x, y, w, h =  f['box']
            face = image[y:y+h, x:x+w]
            resize = cv2.resize(face, (160, 160))
            face_embedded = get_embedding(resize)
            predict_name, class_probability = face_predict(face_embedded)

            confstr = "{0}%".format(round(class_probability))
            if(class_probability > 70):
                cv2.putText(image, str(predict_name), (x+5,y-5), font, 0.5, (0, 255, 0), 2)
                cv2.putText(image, str(confstr), (x + 5, y + h - 5), font,0.5, (0, 255, 0), 2 )
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            elif (class_probability < 50):
                cv2.putText(image, str(predict_name), (x+5,y-5), font, 0.5, (0, 0, 255), 2)
                cv2.putText(image, str(confstr), (x + 5, y + h - 5), font,0.5, (0, 0, 255),2 )
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            else:
                cv2.putText(image, str(predict_name), (x+5,y-5), font, 0.5, (0, 165, 255), 2)
                cv2.putText(image, str(confstr), (x + 5, y + h - 5), font,0.5, (0, 165, 255),2 )
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 165, 255), 2)

        cv2.imshow('Live Recognition', image)
        if (cv2.waitKey(1) == ord('q')):
            break
    
    # print("Attendance Successful")
    cam.release()
    cv2.destroyAllWindows()

# recognize_attendence()