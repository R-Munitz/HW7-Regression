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

# (you will probably need to import more things here)

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
y = df['NSCLC'].values # 1 = NSCLC and 0 = small cell 

#initialize model
log_reg_model = logreg.LogisticRegressor(num_feats=X.shape[1])


def test_prediction():
	#assert that predictions are within valid range 	[0,1]
	#may have to add bias col first to X

	y_pred = log_reg_model.make_prediction(X)
	assert np.all(y_pred >= 0) and np.all(y_pred <= 1)

def test_loss_function():
	#asset that loss function returns a + scalar 

	y_pred = log_reg_model.make_prediction(X)
	loss = log_reg_model.loss_function(y, y_pred)
	assert loss >= 0
	assert isinstance(loss, float)

	

def test_gradient():
	#assert that gradient is of same shape as weights
	gradient = log_reg_model.calculate_gradient(y, X)
	assert gradient.shape == log_reg_model.W.shape
	assert gradient.shape == (X.shape[1] + 1, 1) #+1 for bias term
	

def test_training():
	#assert that loss history is not empty after training
	log_reg_model.train_model(X, y)
	assert len(log_reg_model.loss_hist_train) > 0

	pass