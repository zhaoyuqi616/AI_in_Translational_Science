# load the haberman dataset and summarize the shape
from pandas import read_csv
from matplotlib import pyplot
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from numpy import mean
from numpy import std
import dill
import warnings
warnings.filterwarnings("ignore")

# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv'
# load the dataset

columns = ['age', 'year', 'nodes', 'class']
# load the csv file as a data frame
dataframe = read_csv(url, header=None, names=columns)
print(dataframe.shape)
# summarize the class distribution
target = dataframe['class'].values
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))

# show summary statistics
print(dataframe.describe())
# plot histograms
dataframe.hist()
pyplot.show()
# Multilayer Perceptron (MLP) model

# ensure all data are floating point values
X, y = dataframe.values[:, :-1], dataframe.values[:, -1]

X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# prepare cross validation
kfold = StratifiedKFold(10, random_state=1,shuffle=True)
# enumerate splits
scores = list()
for train_ix, test_ix in kfold.split(X, y):
	# split data
	X_train, X_test, y_train, y_test = X[train_ix], X[test_ix], y[train_ix], y[test_ix]
	# determine the number of input features
	n_features = X.shape[1]
	# define model
	model = Sequential()
	model.add(Dense(100, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# compile the model
	model.compile(optimizer='adam', loss='binary_crossentropy')
	# fit the model
	model.fit(X_train, y_train, epochs=1000, batch_size=16, verbose=0)
	# predict test set
	yhat = model.predict_classes(X_test)
	# evaluate predictions
	score = accuracy_score(y_test, yhat)
	print('>%.3f' % score)
	scores.append(score)
# summarize all scores
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
#dill.dump_session('./MLP_BRCA_Haberman_Survival.pkl')
#to restore session:
#dill.load_session('./MLP_BRCA_Haberman_Survival.pkl')
