"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
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

# Scale the data, since values vary across feature. Note that we
# fit on the training data and use the same scaler for X_val.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

#add bias term - padding data with vector of ones for bias term
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

#initialize model
log_reg_model = logreg.LogisticRegressor(num_feats=X_train.shape[1])

#train model
log_reg_model.train_model(X_train, y_train, X_val, y_val)


"""
#load dataset
df = pd.read_csv("./data/nsclc.csv")

#choose features
features = [
	'Penicillin V Potassium 500 MG',
	'Computed tomography of chest and abdomen',
	'Plain chest X-ray (procedure)', 
	'Low Density Lipoprotein Cholesterol',
	'Creatinine',
	'AGE_DIAGNOSIS'
]
X = df[features].values
#add bias term - padding data with vector of ones for bias term
#X = np.hstack([X, np.ones((X.shape[0], 1))])
y = df['NSCLC'].values # 1 = NSCLC and 0 = small cell 


#initialize model
log_reg_model = logreg.LogisticRegressor(num_feats=6)

#train model

log_reg_model.train_model(X_train, y_train, X_val, y_val)

"""

def test_prediction():
	#assert that predictions are within valid range 	[0,1]
	

	y_pred = log_reg_model.make_prediction(X_train)
	assert np.all(y_pred >= 0) and np.all(y_pred <= 1)

def test_loss_function():
	#asset that loss function returns a + scalar 

	y_pred = log_reg_model.make_prediction(X_train)
	loss = log_reg_model.loss_function(y_train, y_pred)
	assert loss >= 0
	assert isinstance(loss, float)

	

def test_gradient():
	#assert that gradient is of same shape as weights
	gradient = log_reg_model.calculate_gradient(y_train, X_train)
	assert gradient.shape == log_reg_model.W.shape
	assert gradient.shape == (X_train.shape[1] + 1, 1) #+1 for bias term
	

def test_training():
	#assert that loss history is not empty after training
	log_reg_model.train_model(X_train, y_train, X_val, y_val)
	assert len(log_reg_model.loss_hist_train) > 0

	pass