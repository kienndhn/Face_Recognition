from numpy import load
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from random import choice 
from numpy import expand_dims
import pickle

# develop a classifier for the 5 Celebrity Faces Dataset
# load dataset
data = load('../faceEmbedding-10samples.npz')
X, y= data['arr_0'], data['arr_1']

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