import pytest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler


#load data
X_train, X_val, y_train, y_val = utils.loadDataset(
	features=[
		'Penicillin V Potassium 500 MG',
		'Computed tomography of chest and abdomen',
		'Plain chest X-ray (procedure)',
		'Low Density Lipoprotein Cholesterol',
		'Creatinine',
		'AGE_DIAGNOSIS'
	],
	split_percent=0.8,
	split_seed=42
)

# scale the data, since values vary across feature. Note that we
# fit on the training data and use the same scaler for X_val.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

#initialize model
log_reg_model = logreg.LogisticRegressor(num_feats=6)


def test_prediction():
	#assert that predictions are within valid range 	[0,1]

	#add bias term
	X_test = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

	y_pred = log_reg_model.make_prediction(X_test)
	assert np.all(y_pred >= 0) and np.all(y_pred <= 1)


def test_loss_function():
	#assert that loss function returns a + scalar 

	#add bias term
	X_test = np.hstack([X_train, np.ones((X_train.shape[0], 1))]) 

	y_pred = log_reg_model.make_prediction(X_test)
	loss = log_reg_model.loss_function(y_train, y_pred)
	assert loss >= 0
	assert isinstance(loss, float)


def test_gradient():
	#assert that gradient is of same shape as weights

	#add bias term
	X_test = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

	gradient = log_reg_model.calculate_gradient(y_train, X_test)
	assert gradient.shape == log_reg_model.W.shape
	assert gradient.shape == (X_train.shape[1] + 1) #+1 for bias term
	

def test_training():
	#assert that loss history is not empty after training

	#don't need to add bias term here

	log_reg_model.train_model(X_train, y_train, X_val, y_val)
	assert len(log_reg_model.loss_hist_train) > 0

	