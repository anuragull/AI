#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  stock_predictor.py
#  
#  Copyright 2016 AS <as@yo2>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import matplotlib as mplt
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from scipy import stats 
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import gaussian_process
from sklearn import svm
	
# oython imports
import sys # sys.exc_info
import csv # csv.reader
import traceback # (print_tb)
import argparse 

"""
Python decorator for error reporting 
"""
def error_reporter(func):
	def func_wrapper():
		try:
			func()
		except:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print "*** print_exception:"
			traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

	return func_wrapper

"""
	Class that read the file and return, test training validation set
"""
class Data_Processor:
	def __init__(self):
		self.data = []
	
	def read_data(self,filename):
		try:
			with open(filename, 'rb') as csvfile:
				stock_data = csv.reader(csvfile)
				for rows in stock_data:
					self.data.append(rows)
		except:
			print "Error reading file"
			
	def process_data(self,label_index):
		self.labeled = []
		self.unlabeled  = []
		self.unlabeled_date = []
		
		
		for datum in self.data:
			# skip empty rows
			if datum[0] == '':
				continue
			if datum[label_index] == '':
				self.unlabeled.append(datum[1:])
				self.unlabeled_date.append(datum[0:1][0])
			else:
				self.labeled.append(datum[1:]) 
				
		
		#convert to numpy array
		self.labeled = np.array(self.labeled)
		self.unlabeled  = np.array(self.unlabeled)
""" 
	Class that fits any generic linear model
"""
class Model_fitter:
	def __init__(self,model):
		self.model = model
		
	
	def fit_data(self,features,target):
		self.model.fit(features,target)
	
	def compute_score(self,test_features,test_target):
		self.predicted_value = self.model.predict(test_features)
		self.residual_error = np.mean(( self.predicted_value - test_target) ** 2)
		self.variance_score = self.model.score(test_features, test_target)
		
	def  most_important(self):
		import operator
		index, value = max(enumerate(self.model.coef_), key=operator.itemgetter(1))
		return index,value
		
	def get_prediction(self,features):
		return self.model.predict(features)
		
		
"""
	Function to return the best model 
"""

def get_model(target,features,test_features,test_target):
	"""
		the list of models to try
	"""
	model_names = [
		linear_model.LinearRegression(),		
		linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0]),
		tree.DecisionTreeRegressor(),
		gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1),
		SVR(kernel='linear', C=1e3),
		linear_model.BayesianRidge(),
		linear_model.SGDRegressor()
		]
	
	i = 0
	
	best_model = None
	best_score = 0
	
	for model in model_names:
		mf = Model_fitter(model)
		if i >= 4:
			target=target.ravel()
			test_target = test_target.ravel()
		i+=1
		mf.fit_data(features,target)
		mf.compute_score(test_features,test_target)		
		# choose the model with  least error
		if best_model is None:
			best_model = mf
		elif best_model.residual_error > mf.residual_error:
			best_model = mf
	return best_model 
		
"""
	Function that runs the code
	@ input file : File with data 
	@output_file : filename to write csv 
"""

def driver(input_file,output_file,label_index = 1):
	# class to preprocess data
	dp = Data_Processor()
	dp.read_data(input_file)
	dp.process_data(label_index)
	# set 2/3 for training
	training_size = round((len(dp.labeled) -1) * 2 / 3)
	# rest for test 
	test_size = len(dp.labeled) - 1 - training_size
	# target variable
	target = dp.labeled[1:training_size,0:label_index].astype(np.float)
	# features 
	features = dp.labeled[1:training_size,label_index:].astype(np.float)
	
	test_target = dp.labeled[training_size:,0:label_index].astype(np.float)
	test_features = dp.labeled[training_size:,label_index:].astype(np.float)
	
	best_model = get_model(target,features,test_features,test_target)
	out_data = zip(dp.unlabeled_date,best_model.get_prediction(dp.unlabeled[:,label_index:].astype(np.float)))
	
	# write the output
	output_file
	with open(output_file, 'wb') as csvfile:
		prediction_writer = csv.writer(csvfile, delimiter=',')
		prediction_writer.writerow(['Date','Value'])
		for row in out_data:
			prediction_writer.writerow([row[0],row[1][0]])
	 			
@error_reporter
def main():
	parser= argparse.ArgumentParser("Run Prediction on the data set")
	parser.add_argument('input_file',type=str,help="pass the csv file")
	parser.add_argument('outfile',type=str,help="pass output csv file")
	args = parser.parse_args()
	
	driver(args.input_file,args.outfile)
	

if __name__ == '__main__':
	main()

