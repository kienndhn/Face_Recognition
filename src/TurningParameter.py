from numpy import load
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder,Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from random import choice 
from numpy import expand_dims
import pickle
from sklearn.model_selection import GridSearchCV
 

data = load('faceEmbedding-10samples.npz')
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

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear']}
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
 
# fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)
 
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)
 
# print classification report
print(classification_report(y_test, grid_predictions))