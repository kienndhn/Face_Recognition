import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread


from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import expand_dims, load, savez_compressed, asarray
from mtcnn.mtcnn import MTCNN
from keras.models import load_model

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from random import choice 

import pickle

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

def get_image(path):
    image = Image.open(path)
	# image = Image.fromarray(image)
    face_array = asarray(image)
    # print(face_array.shape)
    return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	index = 0
	for filename in listdir(directory):
		# path
		index += 1
		path = directory + filename
		# print(path)
		# get face
		face = get_image(path)
		# store
		faces.append(face)
		if index == 150:
			break
	return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# print(directory)
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# print(path)
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]


def Face_Embedding():
    X, y = load_dataset('../TrainingImage/')
    model = load_model('../facenet_keras.h5')
    
    X_embedded = list()
    # start = time.time()
    for face_pixels in X:
        embedding = get_embedding(model, face_pixels)
        X_embedded.append(embedding)
    # stop = time.time()
    # print("trung binh: ", (stop - start)/len(X))
    X_embedded = asarray(X_embedded)

    savez_compressed('../faceEmbedding-10samples.npz', X_embedded, y)
    # print(X_embedded.shape)
    return X_embedded, y

def FaceClassification():
    # data = load('../faceEmbedding.npz')
    X, y= Face_Embedding()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=1)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(X_val.shape, y_val.shape)

    print('Dataset: train=%d, test=%d' % (X_train.shape[0], X_test.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    X_train = in_encoder.transform(X_train)
    X_test = in_encoder.transform(X_test)
    X_val = in_encoder.transform(X_val)
    # X_val = in_encoder.
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(y_train)
    y_train = out_encoder.transform(y_train)
    y_test = out_encoder.transform(y_test)
    y_val = out_encoder.transform(y_val)

    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # predict
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)
    yhat_val = model.predict(X_val)
    # score
    score_train = accuracy_score(y_train, yhat_train)
    score_test = accuracy_score(y_test, yhat_test)
    score_val = accuracy_score(y_val, yhat_val)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f, val=%.3f' % (score_train*100, score_test*100, score_val*100))
    pickle.dump(model, open("../SVMModel.sav", 'wb'))
    selection = choice([i for i in range(X_val.shape[0])])
# random_face_pixels = X_val[selection]
    random_face_emb = X_val[selection]
    random_face_class = y_val[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])
    # prediction for the face

    # SVMModel = pickle.load(open("SVMModel.sav", 'rb'))
    samples = expand_dims(random_face_emb, axis=0)
    # print(samples)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)

    # print(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    print('Expected: %s' % random_face_name[0])
# plot for fun
# pyplot.imshow(random_face_pixels)
# title = '%s (%.3f)' % (predict_names[0], class_probability)
# pyplot.title(title)
# pyplot.show()

    pickle.dump(model, open("../SVMModel.sav", 'wb'))

# FaceClassification()
# def TrainImages():
    

    

def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1