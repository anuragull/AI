# Prediction Problem

You are given a dataset  that consists of the (simulated) daily open­to­close changes of a set of 10 stocks: S1, S2,  ... , S10.  Stock S1 trades in the U.S. as part of the S&P 500 Index, while stocks S2, S3,  ... , 
S10 trade in Japan, as part of the Nikkei Index.   Your task is to build a model to forecast S1, as a function of S1, S2,  ... , S10.  

You should build your model using the first 50 rows of the dataset.  The remaining 50 rows of the dataset have values for S1 missing: you will use your model to fill these in.  The fund’s researchers believe that some but not all of the lagged values of the other variables are important in predicting S1.

# Questions

Prediction  for S1 on the test dataset.  Predictions should be submitted in a .csv file with two columns.  The first column is a date and the second column consists of predictions for S1 on that date.  The data should be comma delimited and the file titled “predictions.csv”, and should have a header row with column names “Date” and “Value”. 

# Solution

code : stock_predictor.py
Result : Report_stock_prediction.py

